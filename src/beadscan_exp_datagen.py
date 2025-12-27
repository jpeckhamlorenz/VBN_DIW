import os
from beadscan_processor import BeadScan

# %% user input

folderpath = 'data'
scan_speed = 5.0  # mm/s
print_speed = 10.0  # mm/s

visualize = True
save_vis = True
save_data = True
verbose_prints = True


# %% calculations

file_list = [f for f in os.listdir(folderpath) if f.endswith('.csv') and 'cycle' in f]
tool_list = [f for f in os.listdir(folderpath) if f.endswith('.csv') and 'cycle' not in f]

file_list.sort()
tool_list.sort()

if save_vis:
    visualize = True


# %% running cycles
for filename in file_list:

    pattern = filename.rsplit('_', maxsplit=2)[0]
    cycle = filename.split('_', maxsplit=2)[-1].split('.')[0]
    toolname = [t for t in tool_list if pattern in t][0]

    beadscan = BeadScan(folderpath, filename, toolname, scan_speed, print_speed, verbose_prints=verbose_prints)
    Z_rs, R_rs = beadscan.flatten_ransac(visualize=visualize, save_vis=save_vis)
    toolpath_aligned, toolpath_transform = beadscan.register_toolpath_to_scan(visualize=visualize, save_vis=save_vis)

    scan_points = beadscan.points_flattened[beadscan.valid_mask]  # Use only valid points from flattened scan


    profile_xs, profile_zs, ground_lines, areas = beadscan.get_all_profile_areas(toolpath_aligned, scan_points,
                                                                   visualize=visualize, save_vis=save_vis)

    flowrates, volumes = beadscan.get_flowrates(areas, visualize=visualize, save_vis=save_vis)

    if save_data:
        beadscan.save_results(flowrates=flowrates,
                              # volumes=volumes, areas=areas,
                              # scan_points=scan_points,
                              # profile_xs=profile_xs, profile_zs=profile_zs, ground_lines=ground_lines,
                              output_filename='flowrate' + '_' + pattern + '_' + cycle +'.npz')
