"""Legacy-shaped editor cases injected only by tests, outside the product catalog."""

from tempfile import TemporaryDirectory

from smartsom.config.factory_design import FactoryAuthoring, FactoryDesignFile
from smartsom.studio.persistence import TemplateCatalog, TemplateOrigin, file_digest
from smartsom.studio.window import StudioWindow as ProductWindow
from tests.helpers.historical_geometry import editor_factory


def load_template_file(number):
    return FactoryDesignFile(
        schema="smartsom.factory/v2",
        factory=editor_factory(number),
        authoring=FactoryAuthoring(
            operation_catalog_mode="manual" if number == 3 else "auto"
        ),
    )


class EditorFixtureCatalog(TemplateCatalog):
    def load_builtin_file(self, number):
        if isinstance(number, int) and number not in self.records()["local"]:
            target = self.builtin_path(number)
            return load_template_file(number), TemplateOrigin(
                f"Editor fixture {number}", target, file_digest(target), number
            )
        return super().load_builtin_file(number)


class StudioWindow(ProductWindow):
    """Use explicit legacy feature fixtures for the existing editing regressions."""

    def __init__(self, **kwargs):
        fixture_directory = None
        if "data_dir" not in kwargs:
            fixture_directory = TemporaryDirectory(prefix="smartsom-editor-test-")
            kwargs["data_dir"] = fixture_directory.name
        super().__init__(**kwargs)
        self._fixture_directory = fixture_directory
        self.editor.catalog = EditorFixtureCatalog(self.editor.catalog.directory.parent)

    def new_template(self, number=1):
        return super().new_template(number)
