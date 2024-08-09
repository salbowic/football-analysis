from .team_assigner import TeamAssigner

def assign_player_teams(tracks, video_frames, team_assigner): 
    # Assign team colors based on the first frame
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Assign teams to each player in all frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks['players'][frame_num][player_id]["team"] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    return tracks
