# Pinned to the nixpkgs revision all results were produced with (NixOS 26.05.20260905).
{ pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/6713828a351efa628b025a1adf7f43cbf8597513.tar.gz";
    sha256 = "02j9vsjv3w2rwm42vii74bcbpkp0n752j3hci4jih9kxq83wxp8m";
  }) {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      pandas numpy scikit-learn lightgbm xgboost quantile-forest tqdm pyarrow matplotlib
    ]))
  ];
}
