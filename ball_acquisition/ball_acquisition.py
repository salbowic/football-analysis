from player_ball_assigner import PlayerBallAssigner
import numpy as np

def assign_ball_acquisition(tracks, video_frames, team_assigner):
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    return team_ball_control
