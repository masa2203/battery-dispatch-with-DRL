# Battery Dispatch with Deep Reinforcement Learning

This repo contains the code to the paper: 

*Deep Reinforcement Learning for Economic Battery Dispatch: A Comprehensive Comparison of Algorithms and Experiment Design Choices*
by Manuel Sage & Yaoyao Fiona Zhao.

Published at the Journal of Energy Storage: https://doi.org/10.1016/j.est.2025.115428

### Note:
- The data used in the case studies can be found in the /data folder and the parameters in /environments/env_params.py.
- The discharge behavior is handled slightly different in the code than in Eq. (1) in the paper.
- Since uploading this repo, we have updated and improved our codebase for additional research projects:
    * Battery arbitrage with time-series forecasting and RL:
      * Paper: https://doi.org/10.1115/ES2024-130538
      * Preprint: https://arxiv.org/abs/2410.20005
      * Repository: https://github.com/masa2203/battery_arbitrage_with_drl
    * Joint dispatch of gas turbines and batteries with RL:
      * Preprint: https://dx.doi.org/10.2139/ssrn.5117155
      * Repository: https://github.com/masa2203/joint_bes_gt_dispatch