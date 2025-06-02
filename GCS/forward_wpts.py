import os
import time
import geopandas as gpd
from pymavlink import mavutil

def upload_mission_from_geojson(folder_path, filename="shortest_route.geojson", set_auto_mode=False):
    """
    Uploads a mission to the vehicle from a GeoJSON file containing a LINESTRING route.

    Parameters:
    - folder_path: Path to the folder containing the GeoJSON file.
    - filename: Name of the GeoJSON file. Default is 'shortest_route.geojson'.
    - set_auto_mode: If True, sets the vehicle to AUTO mode after uploading.
    """
    # Load GeoJSON
    file_path = os.path.join(folder_path, filename)
    gdf = gpd.read_file(file_path)
    line = gdf.geometry.iloc[0]
    coords = list(line.coords)
    waypoints = [(lat, lon, 2.0) for lon, lat in coords]

    print("Loaded {} waypoints from {}".format(len(waypoints), file_path))

    # Connect to vehicle
    print("Connecting to vehicle...")
    master = mavutil.mavlink_connection('tcp:127.0.0.1:14550')
    master.wait_heartbeat()
    print("Connected to system (sysid={}, compid={})".format(master.target_system, master.target_component))

    # Clear existing mission
    time.sleep(1)
    master.mav.mission_clear_all_send(master.target_system, master.target_component)
    time.sleep(1)
    print("Cleared existing mission")

    # Send mission count
    master.mav.mission_count_send(master.target_system, master.target_component, len(waypoints))
    print("Sending {} waypoints...".format(len(waypoints)))

    # Respond to mission requests
    for i, (lat, lon, alt) in enumerate(waypoints):
        req = master.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'], blocking=True, timeout=20)
        if req is None or req.seq != i:
            raise RuntimeError("Expected request for seq {}, got {}".format(i, req.seq if req else 'nothing'))

        master.mav.mission_item_int_send(
            master.target_system,
            master.target_component,
            i,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 0, 0, 0,
            int(lat * 1e7),
            int(lon * 1e7),
            alt
        )
        print("Sent waypoint {}/{}".format(i + 1, len(waypoints)))

    # Optional: Set mode to AUTO
    if set_auto_mode:
        print("Setting mode to AUTO...")
        master.set_mode_apm("AUTO")
        time.sleep(1)

    # Wait for acknowledgment
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if ack and ack.type == 0:
        print("Mission upload successful")
    else:
        print("Mission upload failed or incomplete. ACK: {}".format(ack))
