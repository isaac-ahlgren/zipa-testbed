# ZIPA Testbed Project

This repository houses code necessary to perform several Zero Interaction
Pairing and Authentication (ZIPA) schemes. This project also has the option to
passively collect and store data on-board or log it to an NFS server.

All code has been tested to work on the Raspberry Pi 4 Model B.

For a seamless setup and execution of the testbed and its related dependencies,
navigate to the [ZIPA cluster](https://github.com/jc-mart/zipa-cluster)
repository.

## Overview

### Requirements

- Raspberry Pi (Models 4 B or later)
- SD card for flashing PiOS (32 GB or higher recommended)
- Compatible sensors (see below)

### Running the testbed

This segment assumes that the sensors have been hooked up and any necessary
dependencies have been downloaded through the ZIPA cluster project.

By default, the cluster project's Ansible scripts set up the testbed code so
that it runs data collection at the end of its playbook. Noteworthy processes of
the testbed code installation include:

1. Changing the network interface in **main.py** and IP address in
   **zipa_sys.py** to be reflective of the Pi's IP interface and address
1. Compilation of Reed-Solomon error correction C code using the `make` command
   under `lib/rscode-1.3`.

Should the need of manually running the testbed arise, stop the ZIPA testbed
daemon by running `sudo systemctl stop zipa-systemd` to terminate the service
when logged onto the Pi. Next, navigate to the `zipa-testbed` directory and use
the installed `run` command to manually start the testbed.

### Changing modes

Changing from collection to ZIPA protocol mode requires at least two of
everything mentioned in the requirements section. These Pi's also need to be on
the same network to be able to communicate with one another.

To manually change modes, navigate to `src` and edit the `main.py` file. Edit
`collection_mode` to `False`. After changing this across the desired Pi's and on
a computer connected to the same network, edit `zipa_starter.py`. Set `IP_ADDR`
to the computer's IPv4 address and set `TARGET_IP_ADDRESS` to one of the
participating Pi's IPv4 address. In `SELECTED_PROTOCOL`, choose one of the four
available protocols above this field.

Once set, on the Pi's `zipa-testbed` directory, issue the command `run` to run
the testbed. These devices will be on standby for the computer to issue the
command to begin working on the selected ZIPA protocol. On the computer, type
`python3 zipa_starter.py` to begin the protocol.

### Sensors tested

Our testbed used the follwing sensors for data collection:

1. [Adafruit BMP280 barometric pressue and altitude sensor](https://www.adafruit.com/product/2651)
1. [Adafruit SHT31-D temperature and humidity sensor](https://www.adafruit.com/product/2857)
1. [Adafruit VEML7700 lux sensor](https://www.adafruit.com/product/4162)
1. [Generic USB audio card](https://www.adafruit.com/product/1475) with
   [3.5mm jack stereo lavalier microphone](https://www.amazon.com/Microphone-Compatible-Smartphone-Amplifier-Recording/dp/B00VYGVZYO/ref=sr_1_4?crid=13YCV49YJGHM1&dib=eyJ2IjoiMSJ9.UhU-tMqKCWDBIzNfTI7FecC4tncO2zAvXvQ2A7STULhNw_05pJAm1fR6w_qYg_2yXLWd5aJ_b9M18tb76w8Z-UJWhHwsifNLynflHDUBZ95pnB-u3xrJXhGgSMGJyWaETyYsDbrkoOWL2AJ14aGPyKGfM2dyQJxYmzP7CtSd8NFG1ZKHtbnw-zXTvZDE3xcupdBm236WJl1qKUUd2jh6OA5sOphGsoxqdUIkF1AnG6EyJNX5kIW9SpxDPWIVVflSKha3NJBaK5brhrVNsPzROloSDyiKOb2PGLM_8JpN8eQ.c2PwdiPlDfe7Oa6z_KPujBcrYcFv-YKoqGW7A1T27yc&dib_tag=se&keywords=HUACAM+lavalier+lapel+microphone+interview+video&qid=1713551202&s=electronics&sprefix=huacam+lavalier+lapel+microphone+interview+video%2Celectronics%2C95&sr=1-4)
1. [Parallax PIR sensor](https://www.parallax.com/product/pir-sensor-with-led-signal/)
1. [Custom Voltkey sensor](https://dl.acm.org/doi/10.1145/3351251)
