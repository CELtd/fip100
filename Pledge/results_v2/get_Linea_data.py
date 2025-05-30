#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:24:58 2025

@author: juanpablomadrigalcianci
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dune_client.client import DuneClient
dune = DuneClient('M7JZRoVdE85eB8RC1JDlzm3jftS5mUv2')
df=dune.get_latest_result_dataframe(5197425)


