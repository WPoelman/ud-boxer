#!/usr/bin/bash

set -x

black . && isort . && mypy . && pytest