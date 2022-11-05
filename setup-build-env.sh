#!/bin/bash
 
apt-get update
cd ~
 
# Install arduino-cli
apt-get install curl -y
curl -L -o arduino-cli.tar.bz2 https://downloads.arduino.cc/arduino-cli/arduino-cli-latest-linux64.tar.bz2
tar xjf arduino-cli.tar.bz2
rm arduino-cli.tar.bz2
mv `ls -1` /usr/bin/arduino-cli

# Install python, pip and pyserial
apt-get install python3.7 -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
#pip install pyserial

# install libraries and boards
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install arduino:megaavr
arduino-cli lib install "Adafruit Motor Shield V2 Library"
arduino-cli lib install "WiFiNINA"
