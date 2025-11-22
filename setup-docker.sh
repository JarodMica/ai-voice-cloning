#!/bin/bash
function main() {
	uid=${uid:-"$(id -u)"}
	gid=${gid:-"$(id -g)"}
	if [ "$uid" = "0" ]; then # User is running docker as sudo
		uid=1001
	fi
	if [ "$gid" = "0" ]; then # User is running docker as sudo
		gid=1001
	fi
	
    docker build \
        --build-arg UID=$uid \
        --build-arg GID=$gid \
        -t ai-voice-cloning \
        .
}

main
