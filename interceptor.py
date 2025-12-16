import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#        PART 0: THE GLOBAL ARSENAL

JETS = {
    # --- 5TH GEN & NEXT GEN STEALTH ---
    "1": {"name": "F-22 Raptor",       "cr_spd": 400, "max_spd": 750, "g": 9.5, "rcs": 0.0001, "col": "silver"},
    "2": {"name": "F-35 Lightning II", "cr_spd": 300, "max_spd": 540, "g": 9.0, "rcs": 0.005,  "col": "white"},
    "3": {"name": "Su-57 Felon",       "cr_spd": 350, "max_spd": 700, "g": 9.0, "rcs": 0.1,    "col": "cyan"},
    "4": {"name": "J-20 Mighty Dragon","cr_spd": 320, "max_spd": 680, "g": 8.5, "rcs": 0.05,   "col": "yellow"},
    "5": {"name": "YF-23 Black Widow", "cr_spd": 450, "max_spd": 800, "g": 8.5, "rcs": 0.0001, "col": "darkgray"},
    "6": {"name": "Su-75 Checkmate",   "cr_spd": 300, "max_spd": 600, "g": 8.0, "rcs": 0.01,   "col": "lightblue"},

    # --- 4.5 GEN & EURO-DELTAS ---
    "7": {"name": "F-15EX Eagle II",   "cr_spd": 300, "max_spd": 850, "g": 9.0, "rcs": 5.0,    "col": "blue"},
    "8": {"name": "Eurofighter Typhoon","cr_spd":320, "max_spd": 700, "g": 9.0, "rcs": 0.5,    "col": "gray"},
    "9": {"name": "Dassault Rafale",   "cr_spd": 300, "max_spd": 650, "g": 9.5, "rcs": 1.0,    "col": "teal"},
    "10":{"name": "Saab Gripen E",     "cr_spd": 280, "max_spd": 600, "g": 9.0, "rcs": 1.5,    "col": "lightgray"},
    "11":{"name": "Su-35 Flanker-E",   "cr_spd": 310, "max_spd": 750, "g": 10.0,"rcs": 3.0,    "col": "orange"},

    # --- COLD WAR LEGENDS ---
    "12":{"name": "F-14 Tomcat",       "cr_spd": 280, "max_spd": 680, "g": 7.0, "rcs": 8.0,    "col": "darkblue"},
    "13":{"name": "F-4 Phantom II",    "cr_spd": 260, "max_spd": 650, "g": 6.5, "rcs": 25.0,   "col": "olive"},
    "14":{"name": "MiG-25 Foxbat",     "cr_spd": 800, "max_spd": 1000,"g": 4.0, "rcs": 15.0,   "col": "red"},
    "15":{"name": "SR-71 Blackbird",   "cr_spd": 900, "max_spd": 1100,"g": 2.0, "rcs": 10.0,   "col": "orange"},
    "16":{"name": "F-117 Nighthawk",   "cr_spd": 200, "max_spd": 280, "g": 4.0, "rcs": 0.001,  "col": "black"},

    # --- EXPERIMENTAL & HYPERSONIC ---
    "17":{"name": "X-15 Rocket Plane", "cr_spd": 1500,"max_spd": 2200,"g": 6.0, "rcs": 5.0,    "col": "pink"},
    "18":{"name": "Darkstar (Scramjet)","cr_spd":2000,"max_spd": 3000,"g": 3.0, "rcs": 0.01,   "col": "white"},
    "19":{"name": "Su-47 Berkut",      "cr_spd": 300, "max_spd": 700, "g": 10.0,"rcs": 2.0,    "col": "black"},

    # --- BOMBERS & SPECIAL MISSION ---
    "20":{"name": "B-2 Spirit",        "cr_spd": 240, "max_spd": 290, "g": 2.5, "rcs": 0.0001, "col": "darkgrey"},
    "21":{"name": "Tu-160 Blackjack",  "cr_spd": 270, "max_spd": 600, "g": 2.5, "rcs": 30.0,   "col": "white"},
    "22":{"name": "B-52 Stratofortress","cr_spd":240, "max_spd": 280, "g": 2.0, "rcs": 100.0,  "col": "brown"},
    "23":{"name": "AC-130 Gunship",    "cr_spd": 140, "max_spd": 160, "g": 1.5, "rcs": 60.0,   "col": "olive"},
    "24":{"name": "A-10 Warthog",      "cr_spd": 160, "max_spd": 190, "g": 6.0, "rcs": 15.0,   "col": "green"},
}

MISSILES = {
    # --- STRATEGIC AIR DEFENSE (SAM) ---
    "1": {"name": "MIM-104 Patriot", "thrust": 45000, "burn": 20.0, "g": 55.0, "delay": 2.0},
    "2": {"name": "S-400 Triumf",    "thrust": 55000, "burn": 25.0, "g": 40.0, "delay": 1.5},
    "3": {"name": "S-500 Prometheus","thrust": 70000, "burn": 30.0, "g": 35.0, "delay": 2.5},
    "4": {"name": "THAAD",           "thrust": 60000, "burn": 15.0, "g": 45.0, "delay": 1.0},
    "5": {"name": "Aster 30 (SAMP/T)","thrust":40000, "burn": 18.0, "g": 60.0, "delay": 1.0},
    "6": {"name": "David's Sling",   "thrust": 38000, "burn": 20.0, "g": 50.0, "delay": 1.5},

    # --- NAVAL INTERCEPTORS ---
    "7": {"name": "SM-6 (Aegis)",    "thrust": 42000, "burn": 22.0, "g": 50.0, "delay": 1.0},
    "8": {"name": "RIM-162 ESSM",    "thrust": 25000, "burn": 10.0, "g": 50.0, "delay": 0.5},
    "9": {"name": "Sea Ceptor",      "thrust": 18000, "burn": 8.0,  "g": 40.0, "delay": 0.5},

    # --- TACTICAL / SHORT RANGE ---
    "10":{"name": "Iron Dome",       "thrust": 12000, "burn": 15.0, "g": 40.0, "delay": 0.5},
    "11":{"name": "NASAMS (AMRAAM)", "thrust": 18000, "burn": 10.0, "g": 40.0, "delay": 1.0},
    "12":{"name": "Pantsir-S1",      "thrust": 25000, "burn": 6.0,  "g": 35.0, "delay": 0.5},
    "13":{"name": "Vintage SA-2",    "thrust": 20000, "burn": 35.0, "g": 10.0, "delay": 5.0},
    "14":{"name": "FIM-92 Stinger",  "thrust": 8000,  "burn": 4.0,  "g": 20.0, "delay": 0.0},

    # --- AIR TO AIR (Simulated Ground Launch) ---
    "15":{"name": "AIM-9X Sidewinder","thrust":10000, "burn": 5.0,  "g": 60.0, "delay": 0.0},
    "16":{"name": "Meteor (Ramjet)",  "thrust": 15000, "burn": 45.0, "g": 45.0, "delay": 0.0},
    "17":{"name": "R-77 Adder",       "thrust": 16000, "burn": 9.0,  "g": 40.0, "delay": 0.0},
    "18":{"name": "PL-15 (Long Rng)", "thrust": 20000, "burn": 15.0, "g": 30.0, "delay": 0.0},
    "19":{"name": "AIM-54 Phoenix",   "thrust": 25000, "burn": 20.0, "g": 18.0, "delay": 1.0},
}

def user_select_loadout():
    print("\n" + "="*60)
    print("      GLOBAL AIR DEFENSE SIMULATOR - SENSOR FUSION MODE")
    print("="*60)
    print("\n--- TARGET AIRCRAFT DATABASE ---")
    keys = list(JETS.keys())
    for i in range(0, len(keys), 2):
        k1 = keys[i]; v1 = JETS[k1]
        s1 = f"[{k1:>2}] {v1['name']:<18} (Max M{v1['max_spd']/340:.1f})"
        if i+1 < len(keys):
            k2 = keys[i+1]; v2 = JETS[k2]
            s2 = f" | [{k2:>2}] {v2['name']:<18} (Max M{v2['max_spd']/340:.1f})"
        else: s2 = ""
        print(s1 + s2)

    j_choice = input("\n>> SELECT BOGEY (1-10): ")
    if j_choice not in JETS: j_choice = "1"
    
    print("\n--- INTERCEPTOR DATABASE ---")
    keys = list(MISSILES.keys())
    for i in range(0, len(keys), 2):
        k1 = keys[i]; v1 = MISSILES[k1]
        s1 = f"[{k1:>2}] {v1['name']:<18} ({v1['g']}G)"
        if i+1 < len(keys):
            k2 = keys[i+1]; v2 = MISSILES[k2]
            s2 = f" | [{k2:>2}] {v2['name']:<18} ({v2['g']}G)"
        else: s2 = ""
        print(s1 + s2)
        
    m_choice = input("\n>> SELECT BATTERY (1-5): ")
    if m_choice not in MISSILES: m_choice = "1"
    
    return JETS[j_choice], MISSILES[m_choice]

SELECTED_JET, SELECTED_MISSILE = user_select_loadout()


#        PART 1: PHYSICS & MATH


def get_air_density(altitude):
    if altitude < 0: altitude = 0
    return 1.225 * np.exp(-altitude / 8500.0)

def get_mach(velocity, altitude):
    if altitude < 0: altitude = 0
    c = 340.0 - (0.004 * altitude) 
    if c < 295: c = 295 
    return np.linalg.norm(velocity) / c

class KalmanFilter3D:
    def __init__(self, dt, initial_state):
        self.dt = dt
        self.x = initial_state
        self.F = np.eye(6)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.P = np.eye(6) * 500  
        self.R = np.eye(3) * 1000 
        self.Q = np.eye(6) * 0.5  
        self.lock_status = "SEARCHING"

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z_meas):
        if z_meas is None:
            self.lock_status = "LOST (PREDICTING)"
            return 
        self.lock_status = "LOCKED"
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z_meas - (self.H @ self.x)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ self.H)) @ self.P

    def get_prediction_trail(self, steps=30):
        future_path = []
        temp_x = self.x.copy()
        for _ in range(steps):
            temp_x = self.F @ temp_x
            future_path.append(temp_x[:3].flatten())
        return np.array(future_path)


#        PART 2: FLIGHT MODEL


class Debris:
    def __init__(self, pos, vel, color):
        self.pos = pos.copy()
        self.vel = vel.copy() + np.random.normal(0, 50, 3)
        self.active = True
        self.color = color

    def update(self, dt):
        if self.pos[2] < 0: 
            self.pos[2] = 0; self.active = False
            return self.pos
        self.vel[2] += -9.81 * dt 
        self.pos += self.vel * dt
        return self.pos

class FighterTarget:
    def __init__(self, specs):
        self.specs = specs
        start_alt = np.random.randint(4000, 7000)
        start_x = np.random.randint(-11000, -8000)
        self.pos = np.array([float(start_x), 4000.0, float(start_alt)])
        cruise_spd = float(specs['cr_spd'])
        self.vel = np.array([cruise_spd, -100.0, 0.0]) 
        self.maneuver_start_time = np.random.uniform(3.0, 6.0)
        self.afterburner_active = False
        self.time = 0
        self.alive = True
        self.debris_list = []

    def update(self, dt):
        if not self.alive:
            for d in self.debris_list: d.update(dt)
            return self.pos

        self.time += dt
        target_speed = self.specs['cr_spd']
        if self.time > 2.0:
            target_speed = self.specs['max_spd']
            self.afterburner_active = True
        
        current_speed = np.linalg.norm(self.vel)
        if current_speed < target_speed:
            thrust_dir = self.vel / current_speed
            self.vel += thrust_dir * 40.0 * dt 

        if self.time > self.maneuver_start_time:
            turn_rate = self.specs['g'] * 2.0 
            drag_factor = 0.999 
            self.vel *= drag_factor 
            self.vel[2] -= turn_rate * dt 
            if self.pos[2] < 600: 
                self.vel[2] += (turn_rate + 15) * dt 
        
        self.vel[1] += np.sin(self.time) * 12.0 * dt
        self.pos += self.vel * dt
        if self.pos[2] < 50: self.pos[2] = 50
        return self.pos

    def explode(self):
        self.alive = False
        for _ in range(30): 
            self.debris_list.append(Debris(self.pos, self.vel, self.specs['col']))

class InterceptorMissile:
    def __init__(self, specs):
        self.specs = specs
        self.pos = np.array([0.0, 0.0, 0.0]) 
        self.vel = np.array([0.0, 0.0, 10.0]) 
        self.mass = 200.0 
        self.thrust = specs['thrust']
        self.burn_time = specs['burn']
        self.max_g = specs['g'] * 9.81
        self.launch_delay = specs['delay'] + np.random.uniform(0, 1.0)
        self.time_elapsed = 0.0
        self.launched = False
        self.exploded = False
        
    def launch(self, target_est):
        self.vel = np.array([0.0, 0.0, 100.0]) 
        self.launched = True

    def update_guidance(self, target_pos, target_vel, dt):
        if self.exploded: return self.pos, 0.0
        self.time_elapsed += dt
        
        if not self.launched:
            if self.time_elapsed > self.launch_delay: self.launch(target_pos)
            else: return self.pos, 0.0 

        r_vec = target_pos - self.pos
        v_vec = target_vel - self.vel
        r = np.linalg.norm(r_vec)
        v_closing = np.linalg.norm(v_vec)

        omega = np.cross(r_vec, v_vec) / (r**2 + 1e-6)
        N = 5.0 
        lat_accel = N * v_closing * np.linalg.norm(omega)
        if lat_accel > self.max_g: lat_accel = self.max_g
        
        if np.linalg.norm(omega) > 1e-6:
            accel_dir = np.cross(omega, self.vel) 
            accel_dir = accel_dir / np.linalg.norm(accel_dir)
        else: accel_dir = np.zeros(3)

        guidance = accel_dir * lat_accel * self.mass
        fg = np.array([0, 0, -9.81 * self.mass])
        rho = get_air_density(self.pos[2])
        drag = -0.5 * rho * np.linalg.norm(self.vel)**2 * 0.25 * 0.15 * (self.vel/np.linalg.norm(self.vel))
        
        ft = np.array([0.,0.,0.])
        time_since_launch = self.time_elapsed - self.launch_delay
        if self.launched and time_since_launch < self.burn_time:
            ft = (self.vel / np.linalg.norm(self.vel)) * self.thrust
            self.mass -= 1.5 * dt 
            
        accel_total = (fg + drag + ft + guidance) / self.mass
        self.vel += accel_total * dt
        self.pos += self.vel * dt
        return self.pos, lat_accel/9.81


#        PART 3: LAYOUT & VISUALIZATION


def calculate_pk(dist, closing_vel, lock_status):
    if lock_status != "LOCKED": return 0.05 
    score_dist = max(0, (15000 - dist) / 15000) 
    score_vel = min(1.0, closing_vel / 1500)
    raw_pk = (0.6 * score_dist) + (0.4 * score_vel)
    return 1 / (1 + np.exp(-10 * (raw_pk - 0.5)))

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10))

# --- GRID LAYOUT ---
ax_sim = plt.subplot2grid((3, 2), (0, 0), rowspan=2, projection='3d')
ax_stats = plt.subplot2grid((3, 2), (2, 0))
ax_radar = plt.subplot2grid((3, 2), (0, 1), rowspan=2, projection='polar')
ax_pk = plt.subplot2grid((3, 2), (2, 1))

# --- 1. 3D VISUALS (MIXED MODE: LINES + DOTS) ---
ax_sim.set_title(f"TACTICAL INTERCEPT: SENSOR FUSION", color='lime', fontfamily='monospace', fontsize=12)
ax_sim.set_xlabel("X"); ax_sim.set_ylabel("Y"); ax_sim.set_zlabel("ALT")
ax_sim.xaxis.set_pane_color((0, 0, 0, 1.0)); ax_sim.yaxis.set_pane_color((0, 0, 0, 1.0)); ax_sim.zaxis.set_pane_color((0, 0, 0, 1.0))
ax_sim.grid(False)

# Cyber Grid
X_g = np.linspace(-12000, 8000, 35); Y_g = np.linspace(-5000, 8000, 35)
X_g, Y_g = np.meshgrid(X_g, Y_g)
Z_g = 600 * np.sin(X_g/3000) * np.cos(Y_g/3000) 
Z_g = np.maximum(Z_g, 0)
ax_sim.plot_wireframe(X_g, Y_g, Z_g, color='#00ffff', alpha=0.15, linewidth=0.5)
ax_sim.set_xlim(-12000, 8000); ax_sim.set_ylim(-5000, 8000); ax_sim.set_zlim(0, 10000)

# Starfield
num_stars = 100
sx = np.random.uniform(-15000, 15000, num_stars)
sy = np.random.uniform(-15000, 15000, num_stars)
sz = np.random.uniform(5000, 20000, num_stars)
ax_sim.scatter(sx, sy, sz, c='white', s=1, alpha=0.6)

# --- 2. STATS ---
ax_stats.set_facecolor('black'); ax_stats.axis('off')
stats_text = ax_stats.text(0.05, 0.8, "INITIALIZING...", color='#00ff00', fontfamily='monospace', fontsize=12, va='top')

# --- 3. RADAR (WITH RED DOTS) ---
ax_radar.set_facecolor('black'); ax_radar.set_title("PPI RADAR (RAW + TRACK)", color='lime', fontfamily='monospace', fontsize=12)
ax_radar.grid(True, color='green', linestyle=':', alpha=0.6)
ax_radar.set_ylim(0, 15000); ax_radar.set_theta_zero_location("N"); ax_radar.set_theta_direction(-1)
ax_radar.set_yticklabels([])
radar_sweep, = ax_radar.plot([], [], color='lime', linewidth=2, alpha=0.8)

# The "Clean" Track (Blue Dot)
radar_blip_tgt, = ax_radar.plot([], [], 'o', color=SELECTED_JET['col'], markersize=8, label="Tracked Target")
# The "Noisy" Returns (Red Dots on 2D)
radar_meas_scatter = ax_radar.scatter([], [], c='red', s=10, alpha=0.6, label="Raw Returns")

radar_blip_int, = ax_radar.plot([], [], 'x', color='red', markersize=6)

# --- 4. PK GRAPH ---
ax_pk.set_title("KILL PROBABILITY ANALYTICS", color='white', fontsize=12)
ax_pk.set_xlabel("Time (s)", color='gray'); ax_pk.set_ylabel("Probability", color='gray')
ax_pk.set_ylim(-0.1, 1.1); ax_pk.set_xlim(0, 40)
ax_pk.grid(True, color='green', linestyle='--', alpha=0.2)
pk_line, = ax_pk.plot([], [], color='cyan', linewidth=2)

# --- INIT ---
dt = 0.05
steps = 900
target = FighterTarget(SELECTED_JET)
interceptor = InterceptorMissile(SELECTED_MISSILE)
kf = KalmanFilter3D(dt, np.array([[-6000], [4000], [5000], [300], [-100], [0]]))

# 3D Objects
l_tgt, = ax_sim.plot([], [], [], color=SELECTED_JET['col'], linestyle='-', linewidth=2)
l_shadow_tgt, = ax_sim.plot([], [], [], color='gray', linestyle='--', alpha=0.3)
l_int, = ax_sim.plot([], [], [], color='red', linewidth=2)
l_plume, = ax_sim.plot([], [], [], color='orange', linewidth=3, alpha=0.8)
l_shadow_int, = ax_sim.plot([], [], [], color='gray', linestyle='--', alpha=0.3)

# 3D: MIX of Green Line AND Red Dots
l_est, = ax_sim.plot([], [], [], color='#00ff00', linewidth=1, label='KF Track')
scat_meas_3d = ax_sim.scatter([], [], [], c='red', marker='+', s=20, alpha=0.6, label='Raw Meas.')

l_vec, = ax_sim.plot([], [], [], color='white', linewidth=2)
l_ghost, = ax_sim.plot([], [], [], color='magenta', linestyle=':', alpha=0.6)
scat_debris = ax_sim.scatter([], [], [], c='orange', marker='.', s=15)
l_lockbox, = ax_sim.plot([], [], [], color='lime', linewidth=1.5)

# Data Storage
tx, ty, tz = [], [], []
ix, iy, iz = [], [], []
ex, ey, ez = [], [], [] # THIS WAS MISSING BEFORE - NOW FIXED
mx, my, mz = [], [], [] # Measurement Points
mr, mth = [], []        # 2D polar measurements
time_history = []; pk_history = []
rad_ang_t, rad_dist_t = [], []

def update(frame):
    current_time = frame * dt
    
    if interceptor.exploded:
        if not target.alive:
             target.update(dt)
             dx, dy, dz = [], [], []
             for d in target.debris_list:
                 dx.append(d.pos[0]); dy.append(d.pos[1]); dz.append(d.pos[2])
             scat_debris._offsets3d = (dx, dy, dz)
             stats_text.set_text(">>> TARGET DESTROYED <<<\n\nMISSION SUCCESSFUL")
             stats_text.set_color('red')
             time_history.append(current_time); pk_history.append(1.0)
             pk_line.set_data(time_history, pk_history)
             ax_sim.view_init(elev=20, azim=frame/4)
             return

    # Physics
    t_pos = target.update(dt)
    if not interceptor.launched: kf.lock_status = "ACQUIRING..."
    
    # GENERATE NOISY MEASUREMENTS (RED DOTS)
    z_meas = None
    if t_pos[2] > 800: 
        noise_mag = 10.0 / SELECTED_JET['rcs'] 
        z_meas = t_pos + np.random.normal(0, noise_mag, 3)
        
        # Store for 3D plot
        mx.append(z_meas[0]); my.append(z_meas[1]); mz.append(z_meas[2])
        if len(mx) > 40: mx.pop(0); my.pop(0); mz.pop(0)

        # Store for 2D Radar (Convert to Polar)
        r_noise = np.sqrt(z_meas[0]**2 + z_meas[1]**2)
        th_noise = np.arctan2(z_meas[0], z_meas[1])
        mr.append(r_noise); mth.append(th_noise)
        if len(mr) > 40: mr.pop(0); mth.pop(0)
    
    kf.predict()
    if z_meas is not None: kf.update(z_meas.reshape(3,1))
    else: kf.update(None)
    
    e_pos = kf.x[:3].flatten()
    e_vel = kf.x[3:].flatten()
    i_pos, g_load = interceptor.update_guidance(e_pos, e_vel, dt)
    
    dist = np.linalg.norm(t_pos - i_pos)
    if dist < 40.0 and target.alive:
        interceptor.exploded = True
        target.explode()
        print(f"SPLASH ONE! Time: {current_time:.2f}s")

    # Update 3D Lines
    tx.append(t_pos[0]); ty.append(t_pos[1]); tz.append(t_pos[2])
    ix.append(i_pos[0]); iy.append(i_pos[1]); iz.append(i_pos[2])
    ex.append(e_pos[0]); ey.append(e_pos[1]); ez.append(e_pos[2]) # Now works safely
    
    l_tgt.set_data(tx, ty); l_tgt.set_3d_properties(tz)
    l_shadow_tgt.set_data(tx, ty); l_shadow_tgt.set_3d_properties([0]*len(tz))
    l_int.set_data(ix, iy); l_int.set_3d_properties(iz)
    l_shadow_int.set_data(ix, iy); l_shadow_int.set_3d_properties([0]*len(iz))
    if len(ix) > 5 and not interceptor.exploded:
        l_plume.set_data(ix[-5:], iy[-5:])
        l_plume.set_3d_properties(iz[-5:])
    
    # 3D: GREEN LINE + RED DOTS
    if kf.lock_status == "LOCKED": l_est.set_color('#00ff00'); l_est.set_linestyle('-')
    else: l_est.set_color('yellow'); l_est.set_linestyle(':')
    l_est.set_data(ex, ey); l_est.set_3d_properties(ez)
    scat_meas_3d._offsets3d = (mx, my, mz)

    # 3D: Vectors
    vec_end = t_pos + (target.vel / np.linalg.norm(target.vel)) * 1000 
    l_vec.set_data([t_pos[0], vec_end[0]], [t_pos[1], vec_end[1]])
    l_vec.set_3d_properties([t_pos[2], vec_end[2]])

    if kf.lock_status == "LOCKED":
        ghost_path = kf.get_prediction_trail(steps=40) 
        l_ghost.set_data(ghost_path[:,0], ghost_path[:,1])
        l_ghost.set_3d_properties(ghost_path[:,2])
        offset = 500
        box_x = [t_pos[0]-offset, t_pos[0], t_pos[0]+offset, t_pos[0], t_pos[0]-offset]
        box_z = [t_pos[2], t_pos[2]+offset, t_pos[2], t_pos[2]-offset, t_pos[2]]
        l_lockbox.set_data(box_x, [t_pos[1]]*5)
        l_lockbox.set_3d_properties(box_z)
    else:
        l_ghost.set_data([], []); l_ghost.set_3d_properties([])
        l_lockbox.set_data([], []); l_lockbox.set_3d_properties([])

    # 2D RADAR: CLEAN TRACK + RAW RED DOTS
    r_t = np.sqrt(t_pos[0]**2 + t_pos[1]**2)
    theta_t = np.arctan2(t_pos[0], t_pos[1]) 
    r_i = np.sqrt(i_pos[0]**2 + i_pos[1]**2)
    theta_i = np.arctan2(i_pos[0], i_pos[1])
    
    radar_blip_tgt.set_data([theta_t], [r_t])
    if interceptor.launched: radar_blip_int.set_data([theta_i], [r_i])
    radar_meas_scatter.set_offsets(np.c_[mth, mr]) # Scatter plot update

    sweep_angle = -(frame % 100) * (2*np.pi/100)
    radar_sweep.set_data([sweep_angle, sweep_angle], [0, 15000])

    # PK Graph
    pk = calculate_pk(dist, np.linalg.norm(target.vel - interceptor.vel), kf.lock_status)
    if not interceptor.launched: pk = 0.01
    time_history.append(current_time); pk_history.append(pk)
    pk_line.set_data(time_history, pk_history)
    for c in ax_pk.collections: c.remove()
    ax_pk.fill_between(time_history, 0, pk_history, color='cyan', alpha=0.2)
    
    # Text Stats
    tgt_mach = get_mach(target.vel, t_pos[2])
    int_mach = get_mach(interceptor.vel, i_pos[2])
    ab_status = "ON" if target.afterburner_active else "OFF"
    
    info_str = (
        f"STATUS : [{kf.lock_status}]\n"
        f"---------------------------\n"
        f"TARGET : {SELECTED_JET['name']}\n"
        f"WEAPON : {SELECTED_MISSILE['name']}\n"
        f"---------------------------\n"
        f"RANGE  : {dist:05.0f} m\n"
        f"ALT    : {t_pos[2]:05.0f} m\n"
        f"T-SPD  : M {tgt_mach:.2f} (AB:{ab_status})\n"
        f"M-SPD  : M {int_mach:.2f}\n"
        f"G-LOD  : {g_load:.1f} G"
    )
    stats_text.set_text(info_str)
    ax_sim.view_init(elev=20, azim=frame/4 - 45)

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=False)
plt.show()