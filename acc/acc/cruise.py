

def control(speed=0, acceleration=0, car_in_front=200, gap=5, cruise_speed=None, state=None):
        """Adaptive Cruise Control

           speed: Current car speed (m/s)
           acceleration: Current car acceleration (m/s^2)
           gas: last signal sent. Real number.
           brake: last signal sent. Real number.
           car_in_front: distance in meters to the car in front. (m)
           gap: maximum distance to the car in front (m)
           cruise_speed: desired speed, set via the cruise control buttons. (m/s)
           status: a convenience dictionary to keep state between runs.
        """

        if state is None:
            state = dict(K_p=0.15, K_d=0., K_i=0.0003, d_front_prev=100,
                         t_safe=.5, prev_setpoint=0., integral_setpoint=0.,
                         maintaining_distance=False)

        delta_distance = car_in_front - 2 * gap - speed**2 / (2 * 5)

        # if the car ahead does not allow to get to cruise speed
        # use safe following distance as a measure until cruise speed is reached again
        if delta_distance < 0:
            state['maintaining_distance'] = True
        elif speed >= cruise_speed:
            state['maintaining_distance'] = False

        if state['maintaining_distance']:
            # But override it if we are too close to the car in front.
            set_point = delta_distance
        else:
            # if the distance is not too close maintain cruise speed
            set_point = cruise_speed - speed

        control = (state['K_p'] * set_point +
                   state['K_d'] * (set_point - state['prev_setpoint']) +
                   state['K_i'] * state['integral_setpoint'])

        if control > 1:
            control = 1.
        elif control < -1:
            control = -1.

        if control >= 0:
            gas = control
            brake = 0
        if control < 0:
            gas = 0
            brake = -control

        # ------set variables from previous value-----
        state['prev_setpoint'] = set_point
        state['integral_setpoint'] = state['integral_setpoint'] + set_point

        return brake, gas, state
