# Runs python from Nix (shell.nix) if available, else from uv (uv.lock), else from PATH. Override: RUNNER=nix|uv|python
py() {
  local runner=${RUNNER:-}
  if [ -z "$runner" ]; then
    if command -v nix-shell >/dev/null 2>&1; then runner=nix
    elif command -v uv >/dev/null 2>&1; then runner=uv
    else runner=python; fi
  fi
  case $runner in
    nix) nix-shell --run "python $1" ;;
    uv)  eval "uv run python $1" ;;
    *)   eval "python $1" ;;
  esac
}
