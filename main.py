from utils import save_video, read_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner, assign_player_teams
from ball_possesion import assign_ball_possesion
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # Interpolate ball postions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    tracks = assign_player_teams(tracks, video_frames, team_assigner)
    team_colors = team_assigner.team_colors
    
    # Assign ball possesion
    player_assigner = PlayerBallAssigner()
    team_ball_possesion = assign_ball_possesion(tracks, player_assigner)
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_possesion, team_colors)
    
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
if __name__ == "__main__":
    main()