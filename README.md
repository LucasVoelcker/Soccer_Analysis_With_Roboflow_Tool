In this project, I used Roboflow to transform a series of videos into an annotated image dataset.
The classes I annotated were: 'goal', 'ball', and 'player'.
I exported the annotated dataset from Roboflow to my computer and trained the images using YOLOv8.

In this project specifically, I implemented an algorithm that detects the soccer field lines analytically, allowing it to run quickly and in parallel 
with the neural network inference. From the field lines, I can correlate points in the image with a soccer field seen from above and translate the coordinates 
of the players on the field to the top-down view using homography.

To visualize the project working and test it with a video, I developed the code 'server.js', which is a web page that calls the Python scripts that process 
the video. On this page, you can choose a video file from your computer, wait for the video to be processed, and then the web page will show the original video 
with the players' bounding boxes and a map of the field drawn below, showing the position of each player.

Below is a sample of the web page in operation:
![GravaodeTela2026-03-04171306-ezgif com-resize](https://github.com/user-attachments/assets/38a011de-c53a-4866-a5b3-1ea508cb5b53)

P.S. I am still working on improvements to this project, like differentiating players of different teams, keeping the same ID to each player (when they appear in a sequence of frames), and tracking the ball.
