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
        cruise_speeds=[(50 * CV.MPH_TO_MS, 0)]
    ),
    Maneuver(
        'while cruising at 60 mph, change cruise speed to 50mph',
        duration=30.,
        initial_speed=60. * CV.MPH_TO_MS,
        cruise_speeds=[(50 * CV.MPH_TO_MS, 0)]
    ),
    Maneuver(
        'while cruising at 20mph, grade change +10%',
        duration=25.,
        initial_speed=20. * CV.MPH_TO_MS,
        cruise_speeds=[(20. * CV.MPH_TO_MS, 0)],
        grade_values=[0., 0., 1.0],
        grade_breakpoints=[0., 10., 11.]
    ),
    Maneuver(
        'while cruising at 20mph, grade change -10%',
        duration=25.,
        initial_speed=20. * CV.MPH_TO_MS,
        cruise_speeds=[(20. * CV.MPH_TO_MS, 0)],
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
        cruise_speeds=[(60. * CV.MPH_TO_MS, 0)]
    ),
    Maneuver(
        'approaching a 0mph car while cruising at 40mph from 150m away',
        duration=30.,
        initial_speed=40. * CV.MPH_TO_MS,
        lead_relevancy=True,
        initial_distance_lead=150.,
        speed_lead_values=[0. * CV.MPH_TO_MS, 0. * CV.MPH_TO_MS],
        speed_lead_breakpoints=[0., 100.],
        cruise_speeds=[(40. * CV.MPH_TO_MS, 0)]
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
        cruise_speeds=[(20., 0)]
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
        cruise_speeds=[(20., 0)]
    ),
    # giving cruise speed same as initial
    Maneuver(
        'starting at 0mph, approaching a stopped car 100m away',
        duration=30.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=100.,
        cruise_speeds=[(20., 0)]
    ),
    Maneuver(
        "following a car at 60mph, lead accel and decel at 0.5m/s^2 every 2s",
        duration=25.,
        initial_speed=30.,
        lead_relevancy=True,
        initial_distance_lead=49.,
        speed_lead_values=[30., 30., 29., 31., 29., 31., 29.],
        speed_lead_breakpoints=[0., 6., 8., 12., 16., 20., 24.],
        cruise_speeds=[(30., 0)]
    ),
    Maneuver(
        "following a car at 10mph, stop and go at 1m/s2 lead dece1 and accel",
        duration=70.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=20.,
        speed_lead_values=[10., 0., 0., 10., 0., 10.],
        speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
        cruise_speeds=[(10., 0)]
    ),
    # setting cruise speed at random
    Maneuver(
        "green light: stopped behind lead car, lead car accelerates at 1.5 m/s",
        duration=30.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=11.,
        speed_lead_values=[0, 0, 45],
        speed_lead_breakpoints=[0, 10., 40.],
        cruise_speeds=[(30., 0)]
    ),
    Maneuver(
        "stop and go with 1m/s2 lead decel and accel, with full stops",
        duration=70.,
        initial_speed=0.,
        lead_relevancy=True,
        initial_distance_lead=20.,
        speed_lead_values=[10., 0., 0., 10., 0., 0.],
        speed_lead_breakpoints=[10., 20., 30., 40., 50., 60.],
        cruise_speeds=[(30., 0)]
    ),
    Maneuver(
        "accelerate from 20 while lead vehicle decelerates from 40 to 20 at 1m/s2",
        duration=30.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=10.,
        speed_lead_values=[20., 10.],
        speed_lead_breakpoints=[1., 11.],
        cruise_speeds=[(20., 0)]
    ),
    Maneuver(
        "accelerate from 20 while lead vehicle decelerates from 40 to 0 at 2m/s2",
        duration=30.,
        initial_speed=10.,
        lead_relevancy=True,
        initial_distance_lead=20.,
        speed_lead_values=[20., 0.],
        speed_lead_breakpoints=[1., 11.],
        cruise_speeds=[(20., 0)]
    )
]

expected = [None for m in maneuvers]

testdata = zip(maneuvers, expected)


@pytest.mark.parametrize("maneuver,score", testdata, ids=[m.title for m in maneuvers])
def test_maneuvers(maneuver, score):
    verbosity = pytest.config.getoption('verbose')
    plot = False
    animate = False
    if verbosity > 4:
        plot = True
    if verbosity > 5:
        animate = True
    # assertions in evaluate will make tests fail if needed.
    maneuver.evaluate(control=control, plot=plot, animate=animate, verbosity=verbosity)


def test_verbose_run():
    """Runs tests in verbose mode with plotting and all.
    """
    # assertions in evaluate will make tests fail if needed.
    maneuvers[2].evaluate(control=control, verbosity=5, animate=True, plot=True)
