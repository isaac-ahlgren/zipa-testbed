# ZIPA Testbed Project

This repository houses code necessary to perform several Zero Interaction Pairing and Authentication (ZIPA) schemes. This project also has the option to passively collect and store data on-board.

All code has been tested to work on the Raspberry Pi 4 Model B.

For a seamless setup and execution of the testbed and its related dependencies, navigate to the [ZIPA cluster](https://github.com/jc-mart/zipa-cluster)

## Overview

### Requirements

- Raspberry Pi (Models 4 B or later)
- SD card for flashing PiOS (32 GB or higher recommended)
- Compatible sensors (see below)

### Running the testbed

This segment assumes that the sensors have been hooked up and any necessary dependencies have been downloaded through the ZIPA cluster project.

By default, the cluster project's Ansible scripts set up the testbed code so that it runs data collection at the end of its playbook. Noteworthy processes of the testbed code installation include:

1. Changing the network interface in **main.py** and IP address in **zipa_sys.py** to be reflective of the Pi's IP interface and address
2. Compilation of Reed-Solomon error correction C code using the `make` command under the **rscode-1.3** directory.

Should the need of manually running the testbed arise, stop the ZIPA testbed daemon by running `sudo systemctl stop zipa-systemd` to terminate the service. Next, navigate to the **zipa-testbed** directory and use the installed `run` command to manually start the testbed.

### Changing modes

**This section is currently a work in progress.**

### Sensors tested

Our testbed used the follwing sensors for data collection:

1. [Adafruit BMP280 barometric pressue and altitude sensor](https://www.adafruit.com/product/2651)
2. [Adafruit SHT31-D temperature and humidity sensor](https://www.adafruit.com/product/2857)
3. [Adafruit VEML7700 lux sensor](https://www.adafruit.com/product/4162)
4. [Generic USB audio card](https://www.adafruit.com/product/1475) with [3.5mm jack stereo lavalier microphone](https://www.amazon.com/Microphone-Compatible-Smartphone-Amplifier-Recording/dp/B00VYGVZYO/ref=sr_1_4?crid=13YCV49YJGHM1&dib=eyJ2IjoiMSJ9.UhU-tMqKCWDBIzNfTI7FecC4tncO2zAvXvQ2A7STULhNw_05pJAm1fR6w_qYg_2yXLWd5aJ_b9M18tb76w8Z-UJWhHwsifNLynflHDUBZ95pnB-u3xrJXhGgSMGJyWaETyYsDbrkoOWL2AJ14aGPyKGfM2dyQJxYmzP7CtSd8NFG1ZKHtbnw-zXTvZDE3xcupdBm236WJl1qKUUd2jh6OA5sOphGsoxqdUIkF1AnG6EyJNX5kIW9SpxDPWIVVflSKha3NJBaK5brhrVNsPzROloSDyiKOb2PGLM_8JpN8eQ.c2PwdiPlDfe7Oa6z_KPujBcrYcFv-YKoqGW7A1T27yc&dib_tag=se&keywords=HUACAM+lavalier+lapel+microphone+interview+video&qid=1713551202&s=electronics&sprefix=huacam+lavalier+lapel+microphone+interview+video%2Celectronics%2C95&sr=1-4)
5. [Parallax PIR sensor](https://www.parallax.com/product/pir-sensor-with-led-signal/)
6. [Custom Voltkey sensor](https://dl.acm.org/doi/10.1145/3351251)
