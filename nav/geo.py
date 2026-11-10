"""Spherical-earth helpers plus the survey grid generator.

READING THIS FILE
Every statement below has a comment above it or at the end of the line explaining what
it does and, where it matters, why it is written that way.

All coordinates are (latitude, longitude) tuples in decimal degrees, WGS-84.
All distances are metres. All bearings are degrees clockwise from true north.

At beach scale (< 1 km) a spherical earth is accurate to a few centimetres, which is
well inside GPS error, so no projection library is needed.
"""
import math                          # only standard-library maths; nothing to install

R = 6378137.0                        # earth's mean radius in metres (WGS-84 semi-major axis)


def distance_m(a, b):
    """Great-circle distance in metres between (lat, lon) tuples."""
    lat1, lon1 = map(math.radians, a)                 # convert point a from degrees to radians
    lat2, lon2 = map(math.radians, b)                 # convert point b from degrees to radians
    dlat = lat2 - lat1                                # north-south difference in radians
    dlon = lon2 - lon1                                # east-west difference in radians
    h = (math.sin(dlat / 2) ** 2                      # haversine term for latitude ...
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)  # ... plus longitude term
    return 2 * R * math.asin(math.sqrt(h))            # invert the haversine to get metres


def bearing_deg(a, b):
    """Initial bearing from a to b, degrees clockwise from true north."""
    lat1, lon1 = map(math.radians, a)                 # point a in radians
    lat2, lon2 = map(math.radians, b)                 # point b in radians
    dlon = lon2 - lon1                                # how far east b is from a
    y = math.sin(dlon) * math.cos(lat2)               # east component of the direction
    x = (math.cos(lat1) * math.sin(lat2)              # north component: ...
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))  # ... latitude correction
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0   # atan2 -> 0..360 degrees


def destination(a, bearing, dist_m):
    """Point dist_m metres from a along bearing."""
    lat1, lon1 = map(math.radians, a)                 # start point in radians
    brg = math.radians(bearing)                       # travel direction in radians
    d = dist_m / R                                    # distance as an angle on the sphere
    lat2 = math.asin(math.sin(lat1) * math.cos(d)     # new latitude: ...
                     + math.cos(lat1) * math.sin(d) * math.cos(brg))  # ... great-circle formula
    lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(d) * math.cos(lat1),  # new longitude ...
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))  # ... via atan2
    return (math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)  # wrap lon to ±180


def offset_m(a, north_m, east_m):
    """Offset a point by local north/east metres."""
    lat = a[0] + math.degrees(north_m / R)            # latitude moves 1 rad per R metres
    lon = a[1] + math.degrees(east_m / (R * math.cos(math.radians(a[0]))))  # east shrinks with latitude
    return (lat, lon)                                 # back as a (lat, lon) tuple


def angle_diff(target, current):
    """Signed shortest rotation from current to target, -180..+180."""
    return (target - current + 180.0) % 360.0 - 180.0  # wrap so 350->10 is +20, not -340


def scan_grid(p1, p2, lanes, lane_spacing_m, sample_spacing_m=2.0):
    """Boustrophedon grid.

    p1 and p2 define the first lane (they are the two corners you mark in the field).
    `lanes` lanes are generated parallel to it, offset 90° to the right, alternating
    direction so the rover never drives back over the same sand.
    """
    lane_bearing = bearing_deg(p1, p2)                # direction of the first lane
    perp = (lane_bearing + 90.0) % 360.0              # 90° right = direction lanes march in
    waypoints = []                                    # the list we will return
    for i in range(max(1, int(lanes))):               # at least one lane, even if caller passes 0
        a = destination(p1, perp, i * lane_spacing_m)  # start of lane i
        b = destination(p2, perp, i * lane_spacing_m)  # end of lane i
        if i % 2:                                     # every second lane ...
            a, b = b, a                               # ... is driven backwards = boustrophedon
        n = max(1, int(round(distance_m(a, b) / sample_spacing_m)))  # how many stops along the lane
        for k in range(n + 1):                        # include both ends
            t = k / n                                 # 0.0 at a, 1.0 at b
            waypoints.append((a[0] + (b[0] - a[0]) * t,       # latitude at this fraction ...
                              a[1] + (b[1] - a[1]) * t))      # ... and longitude at this fraction
    return waypoints                                  # flat list of stops, in driving order


def ground_to_latlon(origin, heading_deg, forward_m, right_m):
    """Rotate a rover-relative offset into a GPS coordinate."""
    h = math.radians(heading_deg)                     # rover heading in radians
    north = forward_m * math.cos(h) - right_m * math.sin(h)   # rotate forward/right into north
    east = forward_m * math.sin(h) + right_m * math.cos(h)    # rotate forward/right into east
    return offset_m(origin, north, east)              # turn the offsets into a real coordinate
