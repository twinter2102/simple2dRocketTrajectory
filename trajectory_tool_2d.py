# Problem with orbital manuevering system (surface and angle of attack too)
# Haven't implemented drag as function of AoA, neither lift
# Needs cleanup, (especially sim function)

# Import statments
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Globvars
R = 6371 * 10 ** 3
G = 6.67408 * 10 ** -11
M = 5.9722 * 10 ** 24
g0 = 9.80665
deg2rad = np.pi / 180
rad2deg = 180 / np.pi

# Atmosphere
density = [1.2985, 0.3639, 0.0880, 0.0132, 0.0020, 0]
alt = [-611, 11019, 20063, 32162, 47350, 86000]
atmosphere = interp1d(alt, density)
flag = True

# Rocket class
class rocket:
    def __init__(self, i_mass, f_mass, T, Isp, Cd):
        self.i_mass = i_mass
        self.f_mass = f_mass
        self.T = T
        self.Isp = Isp
        self.Cd = Cd

    # Acceleration due to gravity
    def acc_g(self, r):
        return (G * M / (np.linalg.norm(r) ** 2)) * -(r / np.linalg.norm(r))

    # Acceleration due to thrust
    def acc_t(self, t,  r, rdot, m):
        if m > self.f_mass:
            phi, throttle = self.pilot(t, r, rdot)
            return (np.array([np.sin(phi), np.cos(phi)]) * self.T * throttle / m)
        return np.array([0, 0])

    # Acceleration due to drag CHANGE TO AERO
    def acc_d(self, t, r, rdot, m):
        alt = np.linalg.norm(r) - R
        phi = self.pilot(t, r, rdot)[0]
        #print(self.get_AoA(phi, r, rdot))
        if alt > -611 and alt < 86e3:
            drag = -(self.Cd * (1/2) * atmosphere(alt) * np.linalg.norm(rdot) * rdot) / m
            return drag
        return np.array([0, 0])

    def get_AoA(self, phi, r, rdot):
        u = np.array([0, 1])
        if r[0] >= 0:
            theta_r = np.arccos(np.dot(r, u) / np.linalg.norm(r))
        else:
            theta_r = (2 * np.pi) - np.arccos(np.dot(r, u) / np.linalg.norm(r))
        if rdot[0] >= 0:
            theta_rdot = np.arccos(np.dot(rdot, u) / np.linalg.norm(rdot))
        else:
            theta_rdot = (2 * np.pi) - np.arccos(np.dot(rdot, u) / np.linalg.norm(rdot))
        if phi >= theta_r:
            return ((theta_rdot - phi)) * rad2deg
        else:
            return ((phi - theta_rdot)) * rad2deg

    # The guy flying the thing
    def pilot(self, t, r, rdot):
        if t < 910 or (t > 2450 and t < 2480):
            throttle = 1
        else:
            throttle = 0
        if np.linalg.norm(r) - R < 30e3:
            phi = self.surface2phi(2 * deg2rad, r)
        elif np.linalg.norm(r) - R < 100e3:
            phi = self.surface2phi(30 * deg2rad, r)
        elif t < 910:
            phi = self.surface2phi(60 * deg2rad, r)
        else:
            phi = self.orbital2phi(0 * deg2rad, r)
        return phi, throttle

    def surface2phi(self, radial, r):
        u = np.array([0, 1])
        if r[0] > 0:
            return np.arccos(np.dot(u, r) / (np.linalg.norm(r))) + radial
        else:
            return 2 * np.pi - np.arccos(np.dot(u, r) / (np.linalg.norm(r))) + radial

    def orbital2phi(self, radial, rdot):
        u = np.array([0, 1])
        if rdot[1] >= 0:
            theta_rdot = np.arccos(np.dot(rdot, u) / np.linalg.norm(rdot)) + (np.pi / 2)
        else:
            theta_rdot = (2 * np.pi) - np.arccos(np.dot(rdot, u) / np.linalg.norm(rdot))
        return theta_rdot % (np.pi)

    # Net acceleration due to all forces
    def acc(self, sv, t):
        rdot = sv[0:2]
        r = sv[2:4]
        mass = sv[4]
        throttle = self.pilot(t, r, rdot)[1]
        if mass > self.f_mass:
            mdot = - ((self.T * throttle) / (self.Isp * g0))
        else:
            mdot = 0

        if np.linalg.norm(r) < R:
            return np.array([0, 0, 0, 0, 0])

        rdotdot = self.acc_g(r) + self.acc_t(t, r, rdot, mass) + self.acc_d(t, r, rdot, mass)
        return np.array([rdotdot[0], rdotdot[1], rdot[0], rdot[1], mdot])

    # Simulation
    def sim(self, time_domain, sv0, an):
        nums = odeint(self.acc, sv0, time_domain, hmax=1)
        rs = np.array([nums[:,2], nums[:,3]])
        rdots = np.array([nums[:,0], nums[:,1]])
        masses = np.array(nums[:,4])
        speeds = np.array([np.linalg.norm(rdot) / 1e3 for rdot in rdots.T])
        gravs = np.array([np.linalg.norm(self.acc_g(r)) / g0 for r in rs.T])
        alts = np.array([(np.linalg.norm(r) - R) / 1e3 for r in rs.T])
        thrusts = np.array([np.linalg.norm(self.acc_t(time_domain[i], rs.T[i], rdots.T[i], masses[i])) / g0 for i in range(len(time_domain))])
        drags = np.array([np.linalg.norm(self.acc_d(time_domain[i], rs.T[i], rdots.T[i], masses[i])) / g0 for i in range(len(time_domain))])
        phis = np.array([self.pilot(time_domain[i], rs.T[i], rdots.T[i])[0] for i in range(len(time_domain))])
        alphas = np.array([self.get_AoA(phis[i], rs.T[i], rdots.T[i]) for i in range(len(time_domain))])

        ax1 = plt.subplot2grid((4, 8), (0, 0), rowspan = 3, colspan = 4, aspect='equal')
        ax2 = plt.subplot2grid((4, 8), (0, 4), rowspan = 1, colspan = 4)
        ax3 = plt.subplot2grid((4, 8), (1, 4), rowspan = 1, colspan = 4)
        ax4 = plt.subplot2grid((4, 8), (2, 4), rowspan = 1, colspan = 4)
        ax5 = plt.subplot2grid((4, 8), (3, 4), rowspan = 1, colspan = 4)
        ax6 = plt.subplot2grid((4, 8), (3, 0), rowspan = 1, colspan = 4)

        earth_circ = plt.Circle((0, 0), R, color = 'b')
        atmo_circ = plt.Circle((0, 0), R + 86e3, color = 'c')
        atmo_rec = plt.Rectangle((0, 0), time_domain[-1], 86, color = 'c')
        ax1.add_patch(atmo_circ)
        ax1.add_patch(earth_circ)
        ax5.add_patch(atmo_rec)

        ax2.grid(True, linestyle='--')
        ax3.grid(True, linestyle='--')
        ax4.grid(True, linestyle='--')
        ax5.grid(True, linestyle='--')

        ax1.set_title('Trajectory')
        ax2.set_title('Mass [kg]')
        ax3.set_title('Speed [km/s]')
        ax4.set_title('Accelerations [g0]')
        ax5.set_title('Altitude [km]')

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        plt.gcf().set_facecolor('#f0f0f0')

        if not an:
            ax1.plot(*rs, color = 'r')
            ax2.plot(time_domain, masses, color='g')
            ax3.plot(time_domain, speeds, color='r')
            ax4.plot(time_domain, gravs, time_domain, thrusts, time_domain, drags)
            ax4.legend(['Gravity', 'Thrust', 'Drag'])
            ax5.plot(time_domain, alts)

            plt.tight_layout()
            plt.show()

        else:
            def animate(i):
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                ax5.clear()
                ax6.clear()

                ax1.add_patch(atmo_circ)
                ax1.add_patch(earth_circ)
                ax5.add_patch(atmo_rec)

                ax2.grid(True, linestyle='--')
                ax3.grid(True, linestyle='--')
                ax4.grid(True, linestyle='--')
                ax5.grid(True, linestyle='--')

                ax1.set_title('Trajectory' + ' (t = {})'.format(round(int(time_domain[i]))))
                ax2.set_title('Mass [kg]')
                ax3.set_title('Speed [km/s]')
                ax4.set_title('Accelerations [g0]')
                ax5.set_title('Altitude [km]')

                ax1.plot(rs[0][:i], rs[1][:i], color = 'r')
                ax2.plot(time_domain[:i], masses[:i], color='g')
                ax3.plot(time_domain[:i], speeds[:i], color='r')
                ax4.plot(time_domain[:i], gravs[:i], time_domain[:i], thrusts[:i], time_domain[:i], drags[:i])
                ax4.legend(['Gravity', 'Thrust', 'Drag'])
                ax5.plot(time_domain[:i], alts[:i])
                ax6.plot(time_domain[:i], alphas[:i])

                xdiff = abs(min(time_domain[:i]) - max(time_domain[:i]))
                ydiff = abs(min(alts[:i]) - max(alts[:i]))
                xpad = xdiff * 0.05
                ypad = ydiff * 0.05
                ax5.set_xbound(min(time_domain[:i]) - xpad, max(time_domain[:i]) + xpad)
                ax5.set_ybound(min(alts[:i]) - ypad, max(alts[:i]) + ypad)

                xdiff = abs(min(rs[0][:i]) - max(rs[0][:i]))
                ydiff = abs(min(rs[1][:i]) - max(rs[1][:i]))
                xpad = xdiff * 0.1
                ypad = ydiff * 0.1

                if xdiff > ydiff:
                    ax1.set_xbound(min(rs[0][:i]) - xpad, max(rs[0][:i]) + xpad)
                    ax1.set_ybound(min(rs[1][:i]) - ((xdiff - ydiff)/2) - xpad, max(rs[1][:i]) + ((xdiff - ydiff)/2) + xpad)
                else:
                    ax1.set_xbound(min(rs[0][:i]) - ((ydiff - xdiff)/2) - ypad, max(rs[0][:i]) + ((ydiff - xdiff)/2) + ypad)
                    ax1.set_ybound(min(rs[1][:i]) - ypad, max(rs[1][:i]) + ypad)

            plt.tight_layout()
            ani = animation.FuncAnimation(plt.gcf(), animate, [i*5 for i in range(len(time_domain)//5)][1:-1], interval=100)
            plt.show()


time_domain = np.linspace(0, 10000, 10000)
delta_glider = rocket(24.9e3, 11.6e3, 2 * 1.6e5, 4e4, 2.7)
sv0 = np.array([0, 0, 1, R, 24.9e3])
delta_glider.sim(time_domain, sv0, True)
