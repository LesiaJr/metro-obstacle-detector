FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-rosbag2-py \
        ros-humble-sensor-msgs-py \
        ros-humble-rosidl-runtime-py \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt 2>/dev/null || true
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY src/ /workspace/src/
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["--help"]