#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'


match = {"all_rounder_in_team": 1.0,
         "avg_team_age": 26.0,
         "bowlers_in_team": 2.0,
         "extra_bowls_opponent": 0,
         "first_selection": "Bowling",
         "match_format": "T20",
         "match_light_type": "Day",
         "max_run_given_1over": 6.0,
         "max_run_scored_1over": 12.0,
         "max_wicket_taken_1over": 1,
         "min_run_given_1over": 0,
         "min_run_scored_1over": 3.0,
         "offshore": "No",
         "opponent": "South Africa",
         "player_highest_run": 45.0,
         "player_highest_wicket": 2,
         "players_scored_zero": 3,
         "season": "Winter"}


response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('India will win the match ')
else:
    print('India will not win the match')
