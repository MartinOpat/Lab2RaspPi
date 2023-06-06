#!/bin/bash
mutt -s "Fresh data from pi" -a $1/*.csv -- $2

