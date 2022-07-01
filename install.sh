#!/usr/bin/bash

# Show intermediate command and exit on failure
set -ex

# Python requirements
pip install -r requirements/requirements.txt

# Install Ocaml
apt-get install opam

# Additional GREW requirements
apt-get install wget m4 unzip librsvg2-bin curl bubblewrap

# Ocaml initialization
opam init
opam switch create 4.13.1 4.13.1
eval $(opam env --switch=4.13.1)

# GREW itself
opam remote add grew "http://opam.grew.fr"
opam install grew grewpy

# Refresh Ocaml to pick initialize GREW
eval $(opam env)
