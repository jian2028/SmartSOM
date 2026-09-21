"""Four-zone factory-design workspace with explicit Browse and Edit modes."""

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smartsom.config.factory_design import load_factory_design_file
from smartsom.domain.factory_design import entity_id
from smartsom.studio.canvas import FactoryScene, FactoryView
from smartsom.studio.controls import keep_exclusive_selection
from smartsom.studio.document import FactoryDocument, blank_design
from smartsom.studio.drawing_state import attach_drawing
from smartsom.studio.editor import StudioEditor
from smartsom.studio.items import COLORS, text_value
from smartsom.studio.persistence import TemplateOrigin
from smartsom.studio.properties import PropertyTree, type_label
from smartsom.studio.workspace_style import (
    STYLE,
    WorkspaceSelector,
    apply_light_palette,
    panel,
)


class NewDesignDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_light_palette(self)
        self.setObjectName("newDesignDialog")
        self.setWindowTitle("New factory design")
        self.resize(510, 400)
        layout = QVBoxLayout(self)
        title = QLabel("Start a factory design")
        title.setStyleSheet("font-size: 20px; font-weight: 600; padding-bottom: 8px;")
        layout.addWidget(title)
        self.blank = QRadioButton("Blank grid  ·  20 × 15 cells")
        self.blank.setObjectName("newBlankOption")
        self.template = QRadioButton("Template 1  ·  4 machines  ·  12 × 12 cells")
        self.template.setObjectName("newTemplateOption")
        self.template_2 = QRadioButton("Template 2  ·  8 machines  ·  42 × 12 cells")
        self.template_2.setObjectName("newTemplate2Option")
        self.from_file = QRadioButton("From an existing factory YAML")
        self.from_file.setObjectName("newFromFileOption")
        for radio in (self.blank, self.template, self.template_2, self.from_file):
            layout.addWidget(radio)
        self.template.setChecked(True)
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setObjectName("newTemplatePath")
        self.path.setPlaceholderText("Choose a factory design file…")
        browse = QPushButton("Browse…")
        browse.setObjectName("newTemplateBrowse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        hint = QLabel(
            "Starts in Browse. Choose Edit to change the design, then Save to choose its destination."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #718591; padding: 10px 0;")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create design")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose factory design", "", "Factory YAML (*.yaml *.yml)"
        )
        if path:
            self.path.setText(path)
            self.from_file.setChecked(True)

    def _accept(self):
        if self.from_file.isChecked() and not self.path.text().strip():
            QMessageBox.information(
                self,
                "Choose a file",
                "Choose an existing factory YAML to use as a template.",
            )
            return
        self.accept()


class StudioWindow(QMainWindow):
    """Factory workspace. Browsing and selection never modify a design."""

    def __init__(self, *, settings=None, data_dir=None):
        super().__init__()
        self.setObjectName("studioWindow")
        self.setWindowTitle("SmartSOM Studio")
        self.resize(1420, 860)
        self.setMinimumSize(1000, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        apply_light_palette(self)
        self.setStyleSheet(STYLE)
        self.documents = []
        self._tree_items = {}
        self._resources = {}
        self._selecting = False
        self._untitled_count = 0
        self._settings = settings
        self.layout_mode = "standard"
        self._layout_states = {}
        self._focus_resources_visible = False
        self._focus_resource_width = 220
        self._read_layout_settings()
        self._build_ui()
        self._build_actions()
        self.editor = StudioEditor(self, data_dir)
        self.set_workspace_layout("standard", remember=False)
        self._document_changed(-1)

    @property
    def current_document(self):
        index = self.tabs.currentIndex()
        return self.documents[index] if 0 <= index < len(self.documents) else None

    @property
    def scene(self):
        return self.current_document.scene if self.current_document else None

    @property
    def view(self):
        return self.current_document.view if self.current_document else None

    def _build_ui(self):
        left, left_layout = panel("RESOURCES")
        self.resource_panel = left
        self.resource_filter = QLineEdit()
        self.resource_filter.setObjectName("resourceFilter")
        self.resource_filter.setPlaceholderText("Find a resource…")
        self.resource_filter.setClearButtonEnabled(True)
        search_row = QWidget()
        search_layout = QHBoxLayout(search_row)
        search_layout.setContentsMargins(10, 0, 10, 10)
        search_layout.addWidget(self.resource_filter)
        left_layout.addWidget(search_row)
        self.resource_tree = QTreeWidget()
        self.resource_tree.setObjectName("resourceTree")
        self.resource_tree.setHeaderHidden(True)
        self.resource_tree.setIndentation(14)
        self.resource_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        left_layout.addWidget(self.resource_tree, 1)
        self.resource_tree.currentItemChanged.connect(self._tree_selected)
        self.resource_filter.textChanged.connect(self._filter_resources)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("documentTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(False)
        self.tabs.currentChanged.connect(self._document_changed)
        self.tabs.tabCloseRequested.connect(self.close_document)

        right = QFrame()
        right.setStyleSheet("QFrame { background: #ffffff; }")
        self.property_panel = right
        self.property_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, right)
        self.property_layout.setContentsMargins(0, 0, 0, 0)
        self.property_layout.setSpacing(0)
        self.property_heading, right_layout = panel("PROPERTIES")
        self.property_layout.addWidget(self.property_heading)
        self.property_title = QLabel("Factory overview")
        self.property_title.setObjectName("propertyTitle")
        self.property_title.setWordWrap(True)
        self.property_title.setStyleSheet(
            "font-size: 17px; font-weight: 600; padding: 2px 12px 4px;"
        )
        right_layout.addWidget(self.property_title)
        self.property_subtitle = QLabel("Select a resource to inspect its design.")
        self.property_subtitle.setWordWrap(True)
        self.property_subtitle.setStyleSheet("color: #7b8c97; padding: 0 12px 12px;")
        right_layout.addWidget(self.property_subtitle)
        self.property_tree = PropertyTree()
        self.property_tree.cell_selected.connect(self._highlight_cell)
        right_layout.addStretch(1)
        self.property_layout.addWidget(self.property_tree, 1)

        horizontal = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = horizontal
        horizontal.setObjectName("workspaceSplitter")
        horizontal.addWidget(left)
        horizontal.addWidget(self.tabs)
        horizontal.addWidget(right)
        horizontal.setSizes([220, 890, 310])
        horizontal.setStretchFactor(1, 1)
        horizontal.setCollapsible(1, False)
        left.setMinimumWidth(160)
        right.setMinimumWidth(230)

        bottom, bottom_layout = panel("DESIGN CHECKS")
        self.issue_panel = bottom
        self.issue_tree = QTreeWidget()
        self.issue_tree.setObjectName("issueTree")
        self.issue_tree.setColumnCount(4)
        self.issue_tree.setHeaderLabels(["LEVEL", "RESOURCE", "CHECK", "DETAIL"])
        self.issue_tree.setRootIsDecorated(False)
        self.issue_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.issue_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.issue_tree.setColumnWidth(0, 100)
        self.issue_tree.setColumnWidth(1, 150)
        self.issue_tree.setColumnWidth(2, 220)
        self.issue_tree.itemClicked.connect(self._issue_selected)
        bottom_layout.addWidget(self.issue_tree, 1)

        self.canvas_splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas_splitter.setObjectName("canvasPropertiesSplitter")
        self.canvas_splitter.addWidget(horizontal)
        self.canvas_splitter.setStretchFactor(0, 1)
        self.canvas_splitter.setCollapsible(0, False)
        vertical = QSplitter(Qt.Orientation.Vertical)
        self.issues_splitter = vertical
        vertical.setObjectName("documentIssuesSplitter")
        vertical.addWidget(self.canvas_splitter)
        vertical.addWidget(bottom)
        vertical.setSizes([610, 160])
        vertical.setStretchFactor(0, 1)
        vertical.setCollapsible(0, False)
        self.setCentralWidget(vertical)
        self.issues_button = QToolButton()
        self.issues_button.setObjectName("issuesButton")
        self.statusBar().addWidget(self.issues_button)
        self.status_message = QLabel("Ready")
        self.statusBar().addWidget(self.status_message, 1)
        self.coordinate_label = QLabel("")
        self.statusBar().addPermanentWidget(self.coordinate_label)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.statusBar().addPermanentWidget(self.zoom_label)

    def _build_actions(self):
        toolbar = QToolBar("Factory workspace", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel(
            "SmartSOM  <span style='font-weight:400;color:#738794'>Studio</span>"
        )
        brand.setStyleSheet("font-size: 18px; font-weight: 700; padding-right: 20px;")
        toolbar.addWidget(brand)
        self.new_action = QAction("New", self)
        self.new_action.setObjectName("newAction")
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self.new_dialog)
        self.open_action = QAction("Open…", self)
        self.open_action.setObjectName("openAction")
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_dialog)
        self.close_action = QAction("Close tab", self)
        self.close_action.setShortcut(QKeySequence.StandardKey.Close)
        self.close_action.triggered.connect(
            lambda: self.close_document(self.tabs.currentIndex())
        )
        for action in (self.new_action, self.open_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        self.layout_actions = {}
        self.layout_group = QActionGroup(self)
        self.layout_group.setExclusive(True)
        for mode, label in (("standard", "Standard"), ("focus", "Focus map")):
            action = QAction(label, self)
            action.setObjectName(f"{mode}LayoutAction")
            action.setCheckable(True)
            action.setChecked(mode == "standard")
            action.setToolTip(
                "Resources, map and properties side by side"
                if mode == "standard"
                else "Wide map with properties below; resources can be expanded"
            )
            action.toggled.connect(
                lambda checked, value=mode: (
                    self.set_workspace_layout(value) if checked else None
                )
            )
            self.layout_group.addAction(action)
            self.layout_actions[mode] = action
            toolbar.addAction(action)
            keep_exclusive_selection(action)
        self.resources_action = QAction("Resources", self)
        self.resources_action.setObjectName("resourcesPanelAction")
        self.resources_action.setCheckable(True)
        self.resources_action.setToolTip("Show or hide the resource list in Focus map")
        self.resources_action.toggled.connect(self._toggle_focus_resources)
        toolbar.addAction(self.resources_action)
        toolbar.addSeparator()
        self.fit_action = QAction("Fit map", self)
        self.fit_action.setObjectName("fitMapAction")
        self.fit_action.setShortcut(QKeySequence("F"))
        self.fit_action.triggered.connect(
            lambda: self.view.fit_map() if self.view else None
        )
        toolbar.addAction(self.fit_action)
        self.layer_actions = {}
        for name in ("grid", "names", "ports"):
            action = QAction(
                "Show numbers" if name == "names" else name.capitalize(), self
            )
            action.setObjectName(f"{name}LayerAction")
            action.setCheckable(True)
            action.setChecked(name != "names")
            action.toggled.connect(
                lambda checked, layer=name: self._set_layer(layer, checked)
            )
            self.layer_actions[name] = action
        toolbar.addAction(self.layer_actions["names"])
        self._update_numbers_action()
        self.issues_action = QAction("Design checks", self)
        self.issues_action.setObjectName("designChecksAction")
        self.issues_action.setCheckable(True)
        self.issues_action.toggled.connect(self.issue_panel.setVisible)
        self.issues_button.setDefaultAction(self.issues_action)
        self.layer_menu = QMenu("Layers", self)
        for action in self.layer_actions.values():
            self.layer_menu.addAction(action)
        layers_button = QToolButton()
        layers_button.setObjectName("layersButton")
        layers_button.setText("Layers")
        layers_button.setMenu(self.layer_menu)
        layers_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(layers_button)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Bindings"))
        self.binding_combo = WorkspaceSelector()
        self.binding_combo.setObjectName("bindingModeCombo")
        for label, value in (
            ("Selected", "selected"),
            ("All", "all"),
            ("None", "none"),
        ):
            self.binding_combo.addItem(label, value)
        self.binding_combo.currentIndexChanged.connect(self._bindings_changed)
        toolbar.addWidget(self.binding_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        file_menu = self.menuBar().addMenu("File")
        for action in (self.new_action, self.open_action, self.close_action):
            file_menu.addAction(action)
        view_menu = self.menuBar().addMenu("View")
        for action in self.layout_actions.values():
            view_menu.addAction(action)
        view_menu.addAction(self.resources_action)
        view_menu.addAction(self.issues_action)
        view_menu.addSeparator()
        view_menu.addAction(self.fit_action)
        view_menu.addSeparator()
        for action in self.layer_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        self.appearance_action = QAction("Appearance preview…", self)
        self.appearance_action.setObjectName("appearancePreviewAction")
        self.appearance_action.triggered.connect(self._show_appearance_preview)
        view_menu.addAction(self.appearance_action)

    def _show_appearance_preview(self):
        from smartsom.studio.appearance import AppearanceDialog
        from smartsom.studio.templates import load_template_1

        design = (
            self.current_document.design if self.current_document else load_template_1()
        )
        dialog = AppearanceDialog(design, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _read_layout_settings(self):
        if self._settings is None:
            return
        for mode in ("standard", "focus"):
            prefix = f"workspace/{mode}"
            self._layout_states[mode] = {
                name: self._settings.value(f"{prefix}/{name}")
                for name in ("workspace", "canvas", "issues")
            }
            self._layout_states[mode]["checks_visible"] = self._settings.value(
                f"{prefix}/checks_visible", mode == "standard", type=bool
            )
        try:
            self._focus_resource_width = max(
                160, int(self._settings.value("workspace/focus/resource_width", 220))
            )
        except (TypeError, ValueError):
            pass

    def _remember_layout(self):
        if self.layout_mode == "focus" and not self.resource_panel.isHidden():
            self._focus_resource_width = max(160, self.resource_panel.width())
        self._layout_states[self.layout_mode] = {
            "workspace": self.workspace_splitter.saveState(),
            "canvas": self.canvas_splitter.saveState(),
            "issues": self.issues_splitter.saveState(),
            "checks_visible": self.issues_action.isChecked(),
        }
        if self.layout_mode == "focus":
            self._layout_states["focus"]["resource_width"] = self._focus_resource_width

    def set_workspace_layout(self, mode, *, remember=True):
        """Rearrange the same widgets without rebuilding documents or selections."""
        mode = mode if mode in ("standard", "focus") else "standard"
        if remember and mode == self.layout_mode:
            return
        if remember:
            self._remember_layout()
        for document in self.documents:
            document.scene.hover_entity()
        self.layout_mode = mode
        focus = mode == "focus"
        self.setUpdatesEnabled(False)
        try:
            if focus:
                self.canvas_splitter.addWidget(self.property_panel)
                self.canvas_splitter.setCollapsible(1, False)
                self.property_layout.setDirection(QBoxLayout.Direction.LeftToRight)
                self.property_heading.setFixedWidth(210)
                self.property_panel.setMinimumHeight(140)
                self.workspace_splitter.setSizes([220, 1100])
                self.canvas_splitter.setSizes([490, 190])
            else:
                self.workspace_splitter.addWidget(self.property_panel)
                self.property_layout.setDirection(QBoxLayout.Direction.TopToBottom)
                self.property_heading.setMinimumWidth(0)
                self.property_heading.setMaximumWidth(16777215)
                self.property_panel.setMinimumHeight(0)
                self.workspace_splitter.setSizes([220, 890, 310])
            self.property_panel.show()
            self.resource_panel.setVisible(not focus or self._focus_resources_visible)
            self.resources_action.setVisible(focus)
            self.resources_action.setChecked(self._focus_resources_visible)
            self.layout_actions[mode].setChecked(True)
            state = self._layout_states.get(mode, {})
            self.issues_action.setChecked(state.get("checks_visible", not focus))
            self.issue_panel.setVisible(self.issues_action.isChecked())
            for name, splitter in (
                ("workspace", self.workspace_splitter),
                ("canvas", self.canvas_splitter),
                ("issues", self.issues_splitter),
            ):
                saved = state.get(name)
                if isinstance(saved, QByteArray):
                    splitter.restoreState(saved)
        finally:
            self.setUpdatesEnabled(True)

    def _toggle_focus_resources(self, visible):
        self._focus_resources_visible = visible
        if self.layout_mode == "focus":
            if not visible:
                self._focus_resource_width = max(160, self.resource_panel.width())
            self.resource_panel.setVisible(visible)
            if visible:
                sizes = self.workspace_splitter.sizes()
                width = min(self._focus_resource_width, max(160, sum(sizes) - 400))
                self.workspace_splitter.setSizes([width, max(400, sum(sizes) - width)])

    def closeEvent(self, event):
        for document in tuple(self.documents):
            if not self.editor.can_close(document):
                event.ignore()
                return
        self.editor.timer.stop()
        for document in self.documents:
            self.editor.detach(document)
        if self._settings is not None:
            self._remember_layout()
            for mode, state in self._layout_states.items():
                for name, value in state.items():
                    if value is not None:
                        self._settings.setValue(f"workspace/{mode}/{name}", value)
            self._settings.sync()
        super().closeEvent(event)

    def open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open factory designs", "", "Factory YAML (*.yaml *.yml)"
        )
        for path in paths:
            self.open_path(path)

    def open_path(self, path, *, show_errors=True):
        path = Path(path).expanduser().resolve()
        for index, document in enumerate(self.documents):
            if document.source_path == path:
                self.tabs.setCurrentIndex(index)
                return document
        try:
            envelope, digest = load_factory_design_file(path)
            design = envelope.factory
        except (OSError, ValueError, TypeError) as exc:
            if show_errors:
                QMessageBox.warning(
                    self, "Cannot open factory design", f"{path.name}\n\n{exc}"
                )
            self.status_message.setText(
                f"Could not open {path.name}. Existing documents are unchanged."
            )
            return None
        return self.add_design(
            design, source_path=path, source_digest=digest, authoring=envelope.authoring
        )

    def add_design(
        self,
        design,
        *,
        source_path=None,
        source_digest=None,
        title=None,
        origin=None,
        authoring=None,
    ):
        if not self.editor.resolve_pending():
            return None
        path = Path(source_path).resolve() if source_path else None
        if title is None:
            if path:
                title = path.name
            else:
                self._untitled_count += 1
                title = f"Untitled {self._untitled_count}"
        document = FactoryDocument(design, path, source_digest, title)
        document.origin = origin
        if authoring is not None:
            document.authoring = authoring
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scene = FactoryScene(design, page)
        attach_drawing(scene, document.authoring.drawing_state)
        view = FactoryView(scene, page)
        document.scene, document.view = scene, view
        layout.addWidget(view, 1)
        footer = QLabel(
            "   Hover for machine group  ·  Click to inspect  ·  Wheel to zoom  ·  Middle/Space-drag to pan"
        )
        footer.setStyleSheet(
            "background: #f8fafb; color: #7e919e; font-size: 11px; padding: 9px 4px;"
        )
        layout.addWidget(footer)
        document.footer = footer
        scene.entity_selected.connect(
            lambda selected, doc=document: (
                self.select_entity(selected) if doc is self.current_document else None
            )
        )
        view.zoom_changed.connect(
            lambda value, doc=document: (
                self.zoom_label.setText(f"{value:.0%}")
                if doc is self.current_document
                else None
            )
        )
        view.cell_hovered.connect(lambda x, y, doc=document: self._show_cell(doc, x, y))
        self.documents.append(document)
        self.editor.attach(document)
        index = self.tabs.addTab(page, title)
        self.tabs.setTabToolTip(
            index, str(path) if path else f"{design.name} · Unsaved design"
        )
        self.tabs.setCurrentIndex(index)
        self._document_changed(index)
        return document

    def new_blank(self):
        return self.add_design(blank_design(self._untitled_count + 1))

    def new_template(self, number=1):
        envelope, origin = self.editor.catalog.load_builtin_file(number)
        return self.add_design(
            envelope.factory, origin=origin, authoring=envelope.authoring
        )

    def new_from_path(self, path):
        path = Path(path).expanduser().resolve()
        envelope, digest = load_factory_design_file(path)
        return self.add_design(
            envelope.factory,
            origin=TemplateOrigin(path.stem, path, digest),
            authoring=envelope.authoring,
        )

    def new_dialog(self):
        dialog = NewDesignDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            if dialog.template.isChecked():
                self.new_template()
            elif dialog.template_2.isChecked():
                self.new_template(2)
            elif dialog.from_file.isChecked():
                self.new_from_path(dialog.path.text())
            else:
                self.new_blank()
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Cannot create design", str(exc))

    def close_document(self, index):
        if not 0 <= index < len(self.documents):
            return
        document = self.documents[index]
        if not self.editor.can_close(document):
            return
        document.interaction.cancel()
        self.editor.detach(document)
        if self.editor.active_document is document:
            self.editor.active_document = None
        page = self.tabs.widget(index)
        self.tabs.blockSignals(True)
        self.documents.pop(index)
        self.tabs.removeTab(index)
        self.tabs.blockSignals(False)
        document.undo_stack.clear()
        document.undo_stack.deleteLater()
        page.deleteLater()
        self._document_changed(self.tabs.currentIndex())

    def _document_changed(self, index):
        if not hasattr(self, "layer_actions"):
            return
        if hasattr(self, "editor") and not self.editor.before_switch(index):
            return
        for opened in self.documents:
            opened.scene.hover_entity()
        document = self.current_document
        self._selecting = True
        self.resource_tree.clear()
        self._tree_items.clear()
        self._resources.clear()
        self.issue_tree.clear()
        self.coordinate_label.clear()
        self._selecting = False
        for action in [
            self.close_action,
            self.fit_action,
            *self.layer_actions.values(),
        ]:
            action.setEnabled(document is not None)
        self.binding_combo.setEnabled(document is not None)
        self._update_numbers_action()
        if document is None:
            self.issues_action.setText("Design checks")
            self.property_title.setText("No factory open")
            self.property_subtitle.setText(
                "Open a factory YAML or create a design to begin."
            )
            self.property_tree.clear()
            self.status_message.setText(
                "Ready · Open a factory design to inspect its layout."
            )
            self.setWindowTitle("SmartSOM Studio")
            self.editor.present_selection()
            return
        design = document.design
        self.issues_action.setText(f"Design checks · {len(document.issues)} issues")
        root = QTreeWidgetItem(self.resource_tree, [design.name])
        root.setExpanded(True)
        # None denotes the overview, outside the resource ID namespace.
        self._tree_items[None] = root
        self._resources[None] = design
        collections = [
            ("Machines", "machine", design.machines),
            ("Buffers", "buffer", design.buffers),
            ("Inspection", "inspection", design.inspection_stations),
            ("Scrap bins", "scrap", design.scrap_bins),
            ("Chargers", "charger", design.chargers),
            ("Ports", "port", design.ports),
            ("AGVs", "agv", design.agvs),
        ]
        for label, kind, resources in collections:
            group = QTreeWidgetItem(root, [f"{label}  {len(resources)}"])
            group.setExpanded(kind in ("machine", "agv"))
            for resource in resources:
                identifier = entity_id(resource)
                item = QTreeWidgetItem(group, [resource.name or identifier])
                item.setForeground(0, QColor(COLORS[kind][1]))
                item.setToolTip(0, f"{resource.name}\n{identifier}")
                item.setData(0, Qt.ItemDataRole.UserRole, identifier)
                self._tree_items[identifier] = item
                self._resources[identifier] = resource
        for issue in document.issues:
            severity = text_value(issue.severity)
            item = QTreeWidgetItem(
                self.issue_tree,
                [
                    severity.upper(),
                    issue.entity_id or "Factory",
                    text_value(issue.code),
                    issue.message,
                ],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, issue)
            item.setForeground(
                0, QColor("#a55549" if severity.lower() == "error" else "#9b7634")
            )
            item.setToolTip(3, issue.message)
        if self.issue_tree.topLevelItemCount() == 0:
            QTreeWidgetItem(
                self.issue_tree,
                ["PASS", "Factory", "Design validation", "No design issues found."],
            )
        for name, action in self.layer_actions.items():
            action.blockSignals(True)
            action.setChecked(getattr(document.scene, f"{name}_visible"))
            action.blockSignals(False)
        self.binding_combo.blockSignals(True)
        self.binding_combo.setCurrentIndex(
            self.binding_combo.findData(document.scene.binding_mode)
        )
        self.binding_combo.blockSignals(False)
        self.zoom_label.setText(f"{document.view.transform().m11():.0%}")
        self.status_message.setText(
            f"{design.grid.width} × {design.grid.height} cells  ·  {len(design.machines)} machines  ·  {len(design.ports)} ports  ·  {len(design.agvs)} AGVs  ·  {'Edit' if document.edit_mode else 'Browse'}"
        )
        self.setWindowTitle(f"{document.title} — SmartSOM Studio")
        self._filter_resources(self.resource_filter.text())
        self.editor.selecting_many = document.selected_ids
        try:
            self.select_entity(document.selected_id)
        finally:
            self.editor.selecting_many = None
        self.editor.update_titles(document)

    def select_entity(self, identifier):
        if self._selecting or self.current_document is None:
            return
        if not self.editor.before_select(identifier):
            return
        desired = self.editor.selecting_many
        if desired is None:
            desired = (identifier,) if identifier else ()
        if (
            self.editor.properties.pending
            and not self.editor.refreshing
            and desired == self.current_document.selected_ids
        ):
            return
        self._selecting = True
        try:
            document = self.current_document
            identifier = identifier if identifier in self._resources else None
            document.selected_id = identifier
            document.selected_ids = (
                self.editor.selecting_many
                if self.editor.selecting_many is not None
                else ((identifier,) if identifier else ())
            )
            item = self._tree_items.get(identifier)
            if item:
                self.resource_tree.setCurrentItem(item)
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.resource_tree.scrollToItem(item)
            document.scene.select_entities(document.selected_ids)
            resource = self._resources.get(identifier, document.design)
            display_id = identifier or document.design.factory_id
            self.property_title.setText(resource.name)
            self.property_subtitle.setText(f"{type_label(resource)} · {display_id}")
            self.property_tree.show_resource(resource)
        finally:
            self._selecting = False
        self.editor.present_selection()

    def _tree_selected(self, current, previous):
        if current and not self._selecting:
            identifier = current.data(0, Qt.ItemDataRole.UserRole)
            if identifier is not None or current.parent() is None:
                self.select_entity(identifier)

    def _issue_selected(self, item, column):
        issue = item.data(0, Qt.ItemDataRole.UserRole)
        if issue is not None:
            self.select_entity(issue.entity_id)
            self.property_tree.reveal_field(issue.field)
            if self.current_document.edit_mode:
                self.editor.properties.reveal_field(issue.field)
            resource = self.scene.entity_items.get(issue.entity_id)
            if resource:
                self.view.ensureVisible(resource)

    def _filter_resources(self, text):
        text = text.casefold().strip()

        def filter_item(item):
            if item.childCount():
                matches = [filter_item(item.child(i)) for i in range(item.childCount())]
                visible = any(matches) or text in item.text(0).casefold()
                item.setHidden(not visible)
                if text and visible:
                    item.setExpanded(True)
                return visible
            identifier = item.data(0, Qt.ItemDataRole.UserRole) or ""
            visible = not text or text in f"{item.text(0)} {identifier}".casefold()
            item.setHidden(not visible)
            return visible

        for i in range(self.resource_tree.topLevelItemCount()):
            filter_item(self.resource_tree.topLevelItem(i))

    def _set_layer(self, name, visible):
        if self.scene:
            self.scene.set_layer(name, visible)
        if name == "names":
            self._update_numbers_action()

    def _update_numbers_action(self):
        visible = self.scene is not None and self.scene.names_visible
        action = self.layer_actions["names"]
        action.setText("Hide numbers" if visible else "Show numbers")
        action.setToolTip(
            "Hide all resource numbers on this map"
            if visible
            else "Show all resource numbers on this map"
        )

    def _bindings_changed(self, index):
        if self.scene:
            self.scene.binding_mode = self.binding_combo.currentData()
            self.scene.update_bindings()

    def _highlight_cell(self, cell):
        if self.scene:
            self.scene.highlight_cell(cell)
            self.view.ensureVisible(self.scene.slot_highlight)

    def _show_cell(self, document, x, y):
        if document is self.current_document:
            self.coordinate_label.setText(
                f"Cell ({x}, {y})"
                if 0 <= x < document.design.grid.width
                and 0 <= y < document.design.grid.height
                else ""
            )
