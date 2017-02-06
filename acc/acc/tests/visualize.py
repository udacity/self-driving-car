
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


class CV:
    """ Conversion constants """
    # later can move to a common utilities file to avoid duplication?
    MPH_TO_MS = 1.609 / 3.6
    MS_TO_MPH = 3.6 / 1.609
    KPH_TO_MS = 1. / 3.6
    MS_TO_KPH = 3.6
    MPH_TO_KPH = 1.609
    KPH_TO_MPH = 1. / 1.609
    KNOTS_TO_MS = 1 / 1.9438
    MS_TO_KNOTS = 1.9438


class Visualizer(object):
    """ Visualization class """
    # to use call
    # initialize before the state transition loop:
    #   vis = Visualizer(animate=True, max_speed=100, max_accel=100, max_score=1000)
    # update at the end of iterations of state transition loop:
    #   vis.update_data(cur_time=plant.current_time(), speed=speed, acceleration=acceleration, \
    #            car_in_front=car_in_front, steer_torque=steer_torque, score=neg_score)
    # clean up after the state transition loop (this also makes sure that the plots do not disappear
    # after each maneuver before pressing [Enter] :
    #   vis.show_final_plots()

    def __init__(self, animate, max_speed, max_accel, max_score, show=True):
        # Set the figure parameters
        font = {'size': 9}
        mpl.rc('font', **font)

        self.animate = animate
        fig = plt.figure(num=0, figsize=(15, 8))

        fig.suptitle("ACC Visualization Tool", fontsize=12)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1s = ax1.twinx()
        ax1ss = ax1.twinx()
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0))  # , colspan=2, rowspan=1)
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        ax4s = ax4.twinx()

        # Set titles of subplots
        ax1.set_title('Forward speed and acceleration')
        ax2.set_title('Distance to the car in front')
        ax3.set_title('Controller score')
        ax4.set_title('Gas and brake')
        # set y-limits
        ax1.set_ylim(-max_accel, max_accel)
        ax1s.set_ylim(-max_speed, max_speed)
        ax1ss.set_ylim(-max_speed * CV.MS_TO_MPH, max_speed * CV.MS_TO_MPH)
        ax2.set_ylim(0, 300)
        ax3.set_ylim(0, max_score)
        ax4.set_ylim(0, max_accel)
        ax4s.set_ylim(0, max_accel)

        fig.subplots_adjust(wspace=0.5)
        ax1ss.spines['right'].set_position(('axes', 1.15))
        ax1s.set_frame_on(True)
        ax1s.patch.set_visible(False)
        ax1s.yaxis.set_major_formatter(mpl.ticker.OldScalarFormatter())
        ax1ss.set_frame_on(True)
        ax1ss.patch.set_visible(False)
        ax1ss.yaxis.set_major_formatter(mpl.ticker.OldScalarFormatter())

        # set t-limits
        ax1.set_xlim(0, 5.0)
        ax2.set_xlim(0, 5.0)
        ax3.set_xlim(0, 5.0)
        ax4.set_xlim(0, 5.0)

        # Turn on grids
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        # set label names
        ax1.set_xlabel("time")
        ax1.set_ylabel("acceleration (m/s^2)")
        ax1s.set_ylabel("speed (m/s)")
        ax1ss.set_ylabel("speed (MPH)")

        ax2.set_xlabel("time")
        ax2.set_ylabel("car in front (m)")
        ax3.set_xlabel("time")
        ax3.set_ylabel("controller score")
        ax4.set_xlabel("time")
        ax4.set_ylabel("gas")
        ax4s.set_ylabel("brake")

        # array of time values for plotting purposes
        self.t_values = np.zeros(0)

        # state variables to be plotted
        self.plant_vel = np.zeros(0)
        self.plant_accel = np.zeros(0)
        self.car_in_front = np.zeros(0)
        self.score = np.zeros(0)
        self.gas_control = np.zeros(0)
        self.brake_control = np.zeros(0)

        # set plots
        self.accel_plot, = ax1.plot(
            self.t_values, self.plant_accel, 'b-', label="Acceleration")
        self.vel_plot, = ax1s.plot(self.t_values, self.plant_vel, 'g-',
                                   label="Speed")
        self.car_in_front_plot, = ax2.plot(self.t_values, self.car_in_front, 'b-',
                                           label="Distance to the car in frontr")
        self.score_plot, = ax3.plot(
            self.t_values, self.score, 'b-', label="controller score")
        self.gas_control_plot, = ax4.plot(
            self.t_values, self.gas_control, 'b-', label="gas")
        self.brake_control_plot, = ax4.plot(
            self.t_values, self.brake_control, 'g-', label="brake")

        # set lagends
        ax1.legend([self.vel_plot, self.accel_plot], [self.vel_plot.get_label(),
                                                      self.accel_plot.get_label()])
        ax4.legend([self.gas_control_plot, self.brake_control_plot],
                   [self.gas_control_plot.get_label(), self.brake_control_plot.get_label()])

        # set time limits for sliding
        self.tmin = 0.0
        self.tmax = 5.0
        self.time = 0

        if self.animate:
            plt.ion()
            plt.show()
            plt.pause(0.001)

    def update_data(self, cur_time, speed, acceleration, gas_control, brake_control,
                    car_in_front, steer_torque, score):
        """ called to update the state saved in the visualizer """
        # the plots are updated when sufficient time has passed
        # TODO: add a robust way to make this user-specified
        if cur_time - self.time > 0.05:
            self.time = cur_time
            self.plant_vel = np.append(self.plant_vel, speed)
            self.plant_accel = np.append(self.plant_accel, acceleration)
            self.car_in_front = np.append(self.car_in_front, car_in_front)
            self.score = np.append(self.score, score)
            self.gas_control = np.append(self.gas_control, gas_control)
            self.brake_control = np.append(self.brake_control, brake_control)
            self.t_values = np.append(self.t_values, cur_time)

            self.vel_plot.set_data(self.t_values, self.plant_vel)
            self.accel_plot.set_data(self.t_values, self.plant_accel)
            self.car_in_front_plot.set_data(self.t_values, self.car_in_front)
            self.score_plot.set_data(self.t_values, self.score)
            self.gas_control_plot.set_data(self.t_values, self.gas_control)
            self.brake_control_plot.set_data(self.t_values, self.brake_control)

            # slide plots
            if cur_time >= self.tmax - 1.0:
                self.vel_plot.axes.set_xlim(
                    cur_time - self.tmax + 1.0, cur_time + 1.0)
                self.car_in_front_plot.axes.set_xlim(
                    cur_time - self.tmax + 1.0, cur_time + 1.0)
                self.score_plot.axes.set_xlim(
                    cur_time - self.tmax + 1.0, cur_time + 1.0)
                self.gas_control_plot.axes.set_xlim(
                    cur_time - self.tmax + 1.0, cur_time + 1.0)

            if self.animate:
                plt.show()
                plt.pause(0.001)

    def show_final_plots(self):
        """ show the final plots """
        # plt.ioff()
        plt.show()
        # input("Press [enter] to close the plots.")
        plt.close()
