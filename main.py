from utils import save_video, read_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from ball_acquisition import assign_ball_acquisition

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Interpolate ball postions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
        
    # Assign ball acquisition
    team_ball_control = assign_ball_acquisition(tracks, video_frames, team_assigner)
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
if __name__ == "__main__":
    main()