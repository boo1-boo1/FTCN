{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  languages.python = {
    enable = true;

    lsp.enable = true;
    lsp.package = pkgs.basedpyright;

    uv.enable = true;
    uv.sync.enable = true;

    venv.enable = true;
  };

  packages = with pkgs; [
    zlib
  ];
}
