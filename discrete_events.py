"""
Implementation of a discrete event simulator
Author: Michel Bierlaire
Date: Fri Sep  3 09:39:21 2021
"""

import heapq
from dataclasses import dataclass
import datetime
from time import strftime, gmtime
import numpy as np
from matplotlib import pyplot as plt


@dataclass
class StateVariables:
    """Data class with the state variables
    Time is expressed in minutes
    """

    time: float = 0.0
    nbr_of_arrivals: int = 0
    nbr_of_departures: int = 0
    nbr_of_customers: int = 0
    service_open: bool = True
    green: bool = True
    simulation_completed: bool = False


class EventManager:
    """Object in charge of managing the events' list"""

    def __init__(self, closure):
        self.time_of_next_arrival = 0
        self.time_of_next_departure = None
        self.time_of_closure = closure
        # Events
        self.events = list()

    def nbr_of_events(self):
        """Returns the number of events in the list"""
        return len(self.events)

    def add_event(self, time, event_type, fct):
        """Add an event to the list

        :param time: time tag of the event (in minutes)
        :type time: float

        :param event_type: type of the event. Used only for reporting.
        :type event_type: str

        :param fct: function to be called when the even occurs
        :type fct: function taking time as sole argument
        """

        heapq.heappush(self.events, (time, (event_type, fct)))

    def get_next_event(self):
        """Obtain the next event in chonological order, and remove it
        from the list.
        """
        try:
            time_next_event, item = heapq.heappop(self.events)
            _, fct_next_event = item
            return fct_next_event, time_next_event
        except IndexError:
            return None, None

    def describe_events(self):
        """Generates a string describing the list of events"""
        res = ''
        for key, item in self.events:
            res += f'{key:.2f}\t\t{item[0]}\n'
        return res


class DiscreteEventSimulator:
    """This class implements a simple discrete event simulator with:
    
    - an arrival process of customers that follows a Poisson process,
    - a service time that follows an exponential distribution,
    - a closure time, when no more arrival of customers is admitted,
    - an optional green/red light process, that interrupts the service 
          deterministically.

    To turn off the green/red light process, the parameters green_time
    and red_time must be set to None.

    """

    def __init__(
        self,
        arrival_rate: float,
        service_time: float,
        red_time: float,
        green_time: float,
        closure_time: str,
    ):
        """Ctor

        :param arrival_rate: average number of arrivals per minute
        :type arrival_rate: float

        :param service_time: average service time in minutes
        :type service_time: float

        :param red_time: duration of the "red light" in minutes,
            when there is no service
        :type red_time: float

        :param green_time: duration of the "green light" in minutes,
            when there is service
        :type green_time: float

        :param closure_time: closure time in minutes
        :type closure_time: float

        :raise ValueError: if exactly one of green_time and red_time is None
        """
        self.arrival_rate = arrival_rate
        self.service_time = service_time
        self.red_time = red_time
        self.green_time = green_time
        self.closure_time = closure_time

        if (self.green_time is None) ^ (self.red_time is None):
            error_msg(
                f'Parameters red_time and green_time must be both defined, '
                f'or both set to None. red_time={self.red_time}. '
                f'green_time={self.green_time}'
            )
            raise ValueError(error_msg)
        
        # Event manager
        self.event_manager = EventManager(self.closure_time)

        # State variables
        self.state = StateVariables()

        self.arrivals = {}
        # Statistics to be collected: arrival of each customer

        self.departures = {}
        # Statistics to be collected: departure of each customer

        self.draws_service_time = []
        # Collect the draws generated for the service time. Useful for variance reduction

        self.draws_arrival_time = []
        # Collect the draws generated for the service time. Useful for variance reduction

        self.end_of_operations = None
        # Statistics to be collected: time when operations end

        # Generate the first arrival of a customer
        self.next_arrival()

        # Generate the first red light
        self.next_red_light()

        # Generate the event of the end of simulation
        self.event_manager.add_event(
            time=self.closure_time, event_type='C', fct=self.close_event
        )

    def run(self):
        """Run the simulator by performing all the iterations
        """
        for i in self:
            pass
        
    def next_red_light(self):
        """Create a red light event. t is the current time,
        when the light becomes green.
        """
        if self.green_time is None:
            return
        time_next_red = self.state.time + self.green_time
        if time_next_red <= self.closure_time:
            self.event_manager.add_event(
                time=time_next_red, event_type='-', fct=self.red_event
            )

    def next_green_light(self):
        """Create a green light event. t is the current time,
        when the light becomes red.
        """
        if self.red_time is None:
            return
        time_next_green = self.state.time + self.red_time
        if time_next_green <= self.closure_time:
            self.event_manager.add_event(
                time=time_next_green, event_type='+', fct=self.green_event
            )

    def next_arrival(self):
        """Add the next arrival event, if it happens during the opening
            hours
        """

        if not self.state.service_open:
            return
        nbr_of_minutes = np.random.exponential(1.0 / self.arrival_rate)
        self.draws_arrival_time.append(nbr_of_minutes)
        time_of_arrival = self.state.time + nbr_of_minutes
        if time_of_arrival <= self.closure_time:
            self.event_manager.add_event(
                time=time_of_arrival, event_type='A', fct=self.arrival_event
            )

    def next_departure(self):
        """Add the next departure event, if the light is green"""
        if not self.state.green:
            return
        time_in_service = np.random.exponential(self.service_time)
        self.draws_service_time.append(time_in_service)
        self.event_manager.add_event(
            time=self.state.time + time_in_service,
            event_type='D',
            fct=self.departure_event,
        )

    def red_event(self, t):
        """Updates when the light becomes red

        :param t: current time
        :type t: float
        """
        self.state.time = t
        self.state.green = False
        self.next_green_light()

    def green_event(self, t):
        """Updates when the light becomes green

        :param t: current time
        :type t: float
        """
        self.state.time = t
        self.state.green = True
        self.next_red_light()

        # If there are customers in the queue, service is resumed
        if self.state.nbr_of_customers > 0:
            self.next_departure()

    def arrival_event(self, t):
        """Update state variables upon arrival of a customer

        :param t: current time
        :type t: float
        """

        self.state.time = t
        self.state.nbr_of_arrivals += 1
        self.state.nbr_of_customers += 1
        # Statistics
        self.arrivals[self.state.nbr_of_arrivals] = t

        # Update event list
        self.next_arrival()

        # If there is no queue, the customer is served immediately
        if self.state.nbr_of_customers == 1:
            # Next departure, if there is no queue
            self.next_departure()

    def departure_event(self, t):
        """Update state variables upon departure of a customer

        :param t: current time
        :type t: float
        """
        self.state.time = t
        self.state.nbr_of_departures += 1
        self.state.nbr_of_customers -= 1
        # Statistics
        self.departures[self.state.nbr_of_departures] = t

        # Update event list if the queue is not empty
        if self.state.nbr_of_customers > 0:
            self.next_departure()
        elif not self.state.service_open:
            self.end_of_simulation()

    def close_event(self, t):
        """Update state variables when new customers are not admitted anymore

        :param t: current time
        :type t: float
        """
        self.state.time = t
        self.state.service_open = False
        if self.state.nbr_of_customers == 0:
            self.end_of_simulation()

    def end_of_simulation(self):
        """Update the state variables at the end of simulation.

        :raise RuntimeError: if the list of events is not empty at the end.
        """
        self.end_of_operations = self.state.time
        self.state.simulation_completed = True
        if self.event_manager.nbr_of_events() > 0:
            error_msg = (
                f'List of events not empty at the end of operations: '
                f'{self.event_manager.describe_events()}'
            )
            raise RuntimeError(error_msg)

    def __iter__(self):
        return self

    def __next__(self):
        fct_next_event, time_next_event = self.event_manager.get_next_event()
        if fct_next_event is None:
            if len(self.arrivals) != len(self.departures):
                print(f'{self.state.nbr_of_customers=}')
                print(f'Nbr of arrivals: {len(self.arrivals)}')
                print(f'Nbr of departures: {len(self.departures)}')
                print(f'End of operations: {self.end_of_operations}')
                print(f'{self.state.service_open=}')
                print(f'{self.state.green=}')
            raise StopIteration

        fct_next_event(time_next_event)
        return self

    def get_state(self):
        """
        :return: description of the state variables
        :rtype: str
        """
        return str(self.state)

    def agg_statistics(self):
        """Generate aggregate statistics

        :return: total service time, number of customers
        :rtype: time

        :raise RuntimeError: if there is a discrepancy between the list
            of arrivals and departures
        """

        total = None
        if len(self.arrivals) != len(self.departures):
            keys_in_a = set(self.arrivals.keys())
            keys_in_d = set(self.departures.keys())
            keys_in_a_not_in_d = keys_in_a - keys_in_d
            if len(keys_in_a_not_in_d) > 0:
                report = [
                    f'{self.arrivals[k]:.2f}' for k in keys_in_a_not_in_d
                ]
                error_msg = (
                    f'Customers arriving and not departing: ' f'{report}'
                )
                raise RuntimeError(error_msg)
            keys_in_d_not_in_a = keys_in_d - keys_in_a
            if len(keys_in_d_not_in_a) > 0:
                report = [
                    f'{self.departures[k]:.2f}' for k in keys_in_d_not_in_a
                ]
                error_msg = (
                    f'Customers departing and not arriving: ' f'{report}'
                )
                raise RuntimeError(error_msg)

        for k, arrival in self.arrivals.items():
            departure = self.departures[k]
            duration = departure - arrival
            if total is None:
                total = duration
            else:
                total += duration
        return total, len(self.arrivals)


def queue(n, left, x_min, x_max, per_row):
    """Calculate the coordinate of a customer in the queue.

    :param n: position of the customer in the queue
    :type n: int

    :param left: if True, the queueu is on the left part of the
        picture, and builds on the right. If False, it is the opposite.
    :type left: bool

    :param x_min: coordinate of the left most part of the queue
    :type x_min: float

    :param x_max: coordinate of the right most part of the queue
    :type x_max: float

    :param per_row: number of customers on each row
    :type per_row: int

    :return: coordinates of the customer
    :rtype: tuple(float, float)

    """
    space_between_rows = 0.1

    col = n % per_row
    row = int((n - col) / per_row)
    width = x_max - x_min
    step = width / per_row
    if left:
        x = x_max - col * step
        y = row * space_between_rows
        return x, y
    x = x_min + col * step
    y = row * space_between_rows
    return x, y


def widget(state_index, states):
    """Draw one of the states of the system. It is desgined to be called
    by an interactive widget for Juypter notebook.

    :param state_index: index of the state to be drawn
    :type state_index: int

    :param states: list of all the states
    :type states: list(StateVariables)

    """

    plt.figure(figsize=(9, 9))
    try:
        state = states[state_index]
    except IndexError:
        return
    plt.xlim(-5, 5)
    plt.ylim(0, 2.2)
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_aspect('equal')
    # time
    the_time = strftime(
        '%H:%M:%S',
        gmtime(datetime.timedelta(minutes=state.time).total_seconds()),
    )
    plt.text(-4.5, 0.1, f't={the_time}', ha='left', va='center')
    # Arrivals
    plt.text(-0.15, 0.1, f'A={state.nbr_of_arrivals}', ha='right', va='center')
    # Departures
    plt.text(0.15, 0.1, f'D={state.nbr_of_departures}', ha='left', va='center')
    for i in range(state.nbr_of_departures):
        coord_x, coord_y = queue(i, left=False, x_min=0.5, x_max=4, per_row=40)
        plt.plot(coord_x, coord_y + 1, 'y*')

    # Nbr of customers
    plt.text(
        -0.9, 0.5, f'Queue={state.nbr_of_customers}', ha='left', va='center'
    )
    for i in range(state.nbr_of_customers):
        coord_x, coord_y = queue(
            i, left=True, x_min=-4, x_max=-0.5, per_row=40
        )
        plt.plot(coord_x, coord_y + 1, 'bo')

    # Service open
    if not state.service_open:
        plt.text(-1.1, 0.5, 'No admittance', ha='right', va='center')
        plt.plot((-1.0, -1.0), (0.1, 0.9), linewidth=5, color='red')

    # green
    color = 'green' if state.green else 'red'
    plt.plot((0, 0), (0, 1), linewidth=15, color=color)
    plt.show()
