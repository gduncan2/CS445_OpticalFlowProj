# <span style="color:#a83269"> CS445_OpticalFlowProj </span>
Optical Flow Based Frame Generation Grant Duncan (gduncan2) Soham Kulkarni (sohamsk2) Braden Dean (bdean9) Sharthak Ghosh (sghos52)

### <span style="color:#e36ded"> Project Topic </span>
The project aims to generate intermediate video frames to improve playback and improve user experience. We used optical flow estimation here to estimate motion vectors using the Lucas-Kanade method. The algorithm works by taking two consecutive frames, estimate flow and then warp them to generate an intermediate frame which can then be inserted in between these. 

### <span style="color:#e36ded"> Project Scope </video>
- Pick any video which has some visible stuttering, like a 24 FPS video.
- Extract each frame
- Track key points in the frame
- Estimate optical flow
- Use optical flow vectors to warp a new frame

### <span style="color:#e36ded"> Motivation and Impact </span>
- Artificial frame generation and quality improvement is an active and high impact area research domain. Companies like NVIDIA, AMD and Intel have invested heavily to implement artificial methods to improve 3D graphics and video rendering. Since GPUs have now started to hit an upper limit in terms of processing speed, these newer generations of upscaling and enhancements have become a domain both for depth and breadth.

### <span style="color:#e36ded"> Approach </span>
- We explore one of the earliest practical and influential algorithms i.e. Lucas, B. D., & Kanade, T. (1981) to arrive at our results.
- In order to solve for optical flow, we first need to track "good features". This can be achieved using multiple techniques but over here we have used Shi-Tomansi which builds upon the idea of "Harris corner detection". They key idea is to only take points where both Eigen values are strong.
- Once we get good features to track, we use these points to estimate optical flow. Optical flow estimation gives us the motion vectors which we then use to warp a new frame. 
- For valid flow fields, we interpolate sparse flows to dense fields.
- Finally, we warp to t=0.5 and blend to get an interpolated frame.
The above outlines the basic approach, in our implementation, we feed the new blend back to our optical flow function to further improve the playback rate of a video.
For example, a 30FPS original video is first boosted to a 60FPS blended video(original + artificial), followed by the 60FPS being further boosted to 120FPS(blended + artificial).

### <span style="color:#e36ded"> Results </span>
![RoadRunner](./output.gif)
- The above GIF showcases our algorithm in action at half the original speed. [Frame by frame comparison](https://drive.google.com/file/d/1Y4ir2wgfA-ildlJYmogYe5TJhESbdDk6/view?usp=drive_link) 
- The first video is of the original 30FPS video.
- The stream in the middle shows us the interpolated and original blends.
- The final stream doubles the blended video itself.
- We go from 30 -> 60 -> 120 without any visible drop in quality and the final video generated has no stuttering. [Side by side comparison](https://drive.google.com/file/d/1nAOe7fuyQe0g-la-7g5aTotA02CPsa1m/view?usp=sharing)
<br/>
<br/>
<video controls>
    <source src="combined_comparison_road_runner_x265_super_slow_xstack.mp4" type="video/mp4">
</video>

### <span style="color:#e36ded"> Steps to Run </span>
- Run the notebook sequentially. There is a hard dependency on an NVIDIA GPU, since we are using ```cupy```
- The main driver function is ```interpolate_full_video_parallel```. It takes the path to an input video file and an output.
- Individual frames are generated in three folders prefixed as follows: -
    - ```og```, ```interp``` and ```out```
- The function launches multiple threads in parallel based on the number of available threads
- There are two output videos generated, an interpolated video and the final generated video.
- The interpolated video is where we have stitched all the artificial frames only.
- If running in google collab, ensure that the relevant media drives are mounted and that you enable the T4 gpu for the runtime type.
