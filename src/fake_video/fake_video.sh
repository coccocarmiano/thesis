modprobe v4l2loopback devices=1 video_nr=1 card_label='MGI'
n=$(ls /dev/video* | sort -nr | head -n1 | grep -Eo "[0-9]")
ffmpeg -stream_loop -1 \
       -re -i "data/mauri/videos/labeled.mp4" \
       -vcodec rawvideo \
       -threads 0 \
       -f v4l2 \
       "/dev/video${n}"
