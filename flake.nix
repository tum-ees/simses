{
  description = "Nix Flake for Simses dev environment using uv";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python314
              pkgs.uv
            ];

            shellHook = ''
              unset PYTHONPATH
              export UV_PYTHON=$(which python)
              export UV_PYTHON_DOWNLOADS=never
              uv sync --extra notebooks
              . .venv/bin/activate
              echo "Libraries installed by uv may require dynamically linked dependencies not available on NixOS, they can be patched by running:"
              echo "nix shell github:GuillaumeDesforges/fix-python"
              echo "fix-python --venv .venv"
              echo "alternatively a python installedy by uv can be run using nix-ld without patching"
              date
            '';
          };
        }
      );
    };
}
