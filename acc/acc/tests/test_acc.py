#!/usr/bin/env python
import pytest

from .maneuver import Maneuver
from acc.cruise import control


class CV:
    MPH_TO_MS = 1.609 / 3.6
    MS_TO_MPH = 3.6 / 1.609
    KPH_TO_MS = 1. / 3.6
    MS_TO_KPH = 3.6
    MPH_TO_KPH = 1.609
    KPH_TO_MPH = 1. / 1.609
    KNOTS_TO_MS = 1 / 1.9438
    MS_TO_KNOTS = 1.9438


class CB:
    RES_ACCEL = 4
    DECEL_SET = 3
    CANCEL = 2
    MAIN = 1


maneuvers = [
    Maneuver(
        'while cruising at 40 mph, change cruise speed to 50mph',
        duration=30.,
        initial_speed=40. * CV.MPH_TO_MS,
        cruise_button_presses=[(CB.DECEL_SET, 2.), (0, 2.3),
                               (CB.RES_ACCEL, 10.), (0, 10.1),
                               (CB.RES_ACCEL, 10.2), (0, 10.3)]
    ),
    Maneuver(
        'while cruising at 60 mph, change cruise speed to 50mph',
        duration=30.,
        initial_speed=60. * CV.MPH_TO_MS,
        cruise_button_presses=[(CB.DECEL_SET, 2.), (0, 2.3),
                               (CB.DECEL_SET, 10.), (0, 10.1),
                               (CB.DECEL_SET, 10.2), (0, 10.3)]
    ),
    Maneuver(
        'while cruising at 20mph, grade change +10%',
        duration=25.,
        initial_speed=20. * CV.MPH_TO_MS,
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)],
        grade_values=[0., 0., 1.0],
        grade_breakpoints=[0., 10., 11.]
    ),
    Maneuver(
        'while cruising at 20mph, grade change -10%',
        duration=25.,
        initial_speed=20. * CV.MPH_TO_MS,
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)],
        grade_values=[0., 0., -1.0],
        grade_breakpoints=[0., 10., 11.]
    ),
    Maneuver(
        'approaching a 40mph car while cruising at 60mph from 100m away',
        duration=30.,
        initial_speed=60. * CV.MPH_TO_MS,
        lead_relevancy=True,
        initial_distance_lead=100.,
        speed_lead_values=[40. * CV.MPH_TO_MS, 40. * CV.MPH_TO_MS],
        speed_lead_breakpoints=[0., 100.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)]
    ),
    Maneuver(
        'approaching a 0mph car while cruising at 40mph from 150m away',
        duration=30.,
        initial_speed=40. * CV.MPH_TO_MS,
        lead_relevancy=True,
        initial_distance_lead=150.,
        speed_lead_values=[0. * CV.MPH_TO_MS, 0. * CV.MPH_TO_MS],
        speed_lead_breakpoints=[0., 100.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)]
    ),
    Maneuver(
        'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
        duration=50.,
        initial_speed=20.,
        lead_relevancy=True,
        initial_distance_lead=35.,
        speed_lead_values=[20. * CV.MPH_TO_MS,
                           20. * CV.MPH_TO_MS, 0. * CV.MPH_TO_MS],
        speed_lead_breakpoints=[0., 15., 35.0],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)]
    ),
    Maneuver(
        'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
        duration=50.,
        initial_speed=20.,
        lead_relevancy=True,
        initial_distance_lead=35.,
        speed_lead_values=[20. * CV.MPH_TO_MS,
                           20. * CV.MPH_TO_MS, 0. * CV.MPH_TO_MS],
        speed_lead_breakpoints=[0., 15., 25.0],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3)]
    ),
    Maneuver(
        'starting at 0mph, approaching a stopped car 100m away',
        duration=30.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=100.,
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                               (CB.RES_ACCEL, 1.8), (0.0, 1.9)]
    ),
    Maneuver(
        "following a car at 60mph, lead accel and decel at 0.5m/s^2 every 2s",
        duration=25.,
        initial_speed=30.,
        lead_relevancy=True,
        initial_distance_lead=49.,
        speed_lead_values=[30., 30., 29., 31., 29., 31., 29.],
        speed_lead_breakpoints=[0., 6., 8., 12., 16., 20., 24.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
    ),
    Maneuver(
        "following a car at 10mph, stop and go at 1m/s2 lead dece1 and accel",
        duration=70.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=20.,
        speed_lead_values=[10., 0., 0., 10., 0., 10.],
        speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
    ),
    Maneuver(
        "green light: stopped behind lead car, lead car accelerates at 1.5 m/s",
        duration=30.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=4.,
        speed_lead_values=[0, 0, 45],
        speed_lead_breakpoints=[0, 10., 40.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                               (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                               (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                               (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
    ),
    Maneuver(
        "stop and go with 1m/s2 lead decel and accel, with full stops",
        duration=70.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=20.,
        speed_lead_values=[10., 0., 0., 10., 0., 0.],
        speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7)]
    ),
    Maneuver(
        "accelerate from 20 while lead vehicle decelerates from 40 to 20 at 1m/s2",
        duration=30.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=10.,
        speed_lead_values=[20., 10.],
        speed_lead_breakpoints=[1., 11.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                               (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                               (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                               (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
    ),
    Maneuver(
        "accelerate from 20 while lead vehicle decelerates from 40 to 0 at 2m/s2",
        duration=30.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=10.,
        speed_lead_values=[20., 0.],
        speed_lead_breakpoints=[1., 11.],
        cruise_button_presses=[(CB.DECEL_SET, 1.2), (0, 1.3),
                               (CB.RES_ACCEL, 1.4), (0.0, 1.5),
                               (CB.RES_ACCEL, 1.6), (0.0, 1.7),
                               (CB.RES_ACCEL, 1.8), (0.0, 1.9),
                               (CB.RES_ACCEL, 2.0), (0.0, 2.1),
                               (CB.RES_ACCEL, 2.2), (0.0, 2.3)]
    )
]

MIN_SCORE = 10

# We expect the score to be 0 when it crashes and higher based on how comfortable the ride was.
expected = [MIN_SCORE for m in maneuvers]

testdata = zip(maneuvers, expected)


@pytest.mark.parametrize("maneuver,score", testdata, ids=[m.title for m in maneuvers])
def test_maneuvers(maneuver, score):
    verbosity = pytest.config.getoption('verbose')
    score = maneuver.evaluate(control=control, verbosity=verbosity)

    assert score >= MIN_SCORE

    if verbosity > 0 or True:
        print(maneuver.title, score)
