#!/bin/bash

# WARNING: THIS TAKES A VERY LONG TIME, WOULD NOT RECOMMEND CALLING THIS SCRIPT
python generate_line_summaries.py
python join_lines.py
python generate_gpy_summary.py