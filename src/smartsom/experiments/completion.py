"""Shell integration reuses the public CLI parser and experiment resolver."""

import argparse

from smartsom.config.launch import experiment_names


def candidates(prefix=""):
    from smartsom.experiments.cli import _parser

    names = set(experiment_names())

    def visit(parser):
        for action in parser._actions:
            names.update(action.option_strings)
            if isinstance(action, argparse._SubParsersAction):
                names.update(k for k in action.choices if not k.startswith("_"))
                for child in action.choices.values():
                    visit(child)

    visit(_parser())
    return sorted(name for name in names if name.startswith(prefix))


def script(shell):
    if shell == "bash":
        return """_smartsom_complete() {
  COMPREPLY=()
  local candidate
  while IFS= read -r candidate; do
    COMPREPLY+=("$candidate")
  done < <(smartsom _complete -- "${COMP_WORDS[COMP_CWORD]}")
}
complete -F _smartsom_complete smartsom"""
    return """#compdef smartsom
_smartsom_complete() {
  local -a matches
  matches=("${(@f)$(smartsom _complete -- "${words[CURRENT]}")}")
  compadd -- "${matches[@]}"
}
compdef _smartsom_complete smartsom"""
