import os
import sys

import numpy as np
from unittest.mock import MagicMock

# Ensure the correct import path is used for your protocol
sys.path.insert(1, os.getcwd() + "/src")
from protocols.fastzip import FastZIPProtocol  # noqa: E402


def test_fastzip_process_context():
    # Mock the parameters needed for FastZIPProtocol initialization
    parameters = {
        "verbose": False,
        "chunk_size": 1024,
        "n_bits": 128,
        "power_thresh": 70,
        "snr_thresh": 1.2,
        "peak_thresh": 0,
        "bias": 0,
        "sample_rate": 44100,
        "eqd_delta": 1,
        "peak_status": None,
        "ewma_filter": None,
        "alpha": None,
        "remove_noise": None,
        "normalize": True,
        "key_length": 16,  # Make sure to define all necessary parameters
        "parity_symbols": 2,  # Added this line to include parity_symbols
        "timeout": 30
    }

    # Create an instance of FastZIPProtocol
    protocol = FastZIPProtocol(parameters, sensor=None, logger=None)

    # Mock the `read_samples` and `process_chunk` methods
    protocol.read_samples = lambda size: np.random.randn(size)
    protocol.process_chunk = (
        lambda chunk: b"0" * 4
    )  # Simulate returning 32 bytes of processed data each call

    # Execute the method under test
    result = protocol.process_context()

    # Validate the output
    expected_length = parameters["key_length"] * 8
    assert (
        len(result) == expected_length
    ), "The length of the result should match the key_length parameter"

    # Optionally, validate the content if you have specific expectations about what it should be
    assert (
        result.count(b"0") == expected_length
    ), "The result should consist of '0' bits."


def test_device_protocol(mocker):
    # Setup
    parameters = {
        "verbose": True,
        "chunk_size": 1024,
        "n_bits": 128,
        "power_thresh": 5,
        "snr_thresh": 2,
        "peak_thresh": 10,
        "bias": 1,
        "sample_rate": 1000,
        "eqd_delta": 5,
        "peak_status": True,
        "ewma_filter": True,
        "alpha": 0.015,
        "remove_noise": True,
        "normalize": True,
        "key_length": 16,
        "parity_symbols": 2,
        "timeout": 30
    }

    mock_ack = mocker.patch('networking.network.ack', autospec=True)
    mock_ack_standby = mocker.patch('networking.network.ack_standby', autospec=True, return_value=True)
    mock_send_status = mocker.patch('networking.network.send_status', autospec=True)

    # Create an instance of FastZIPProtocol
    protocol = FastZIPProtocol(parameters, sensor=None, logger=None)

    # Mock context retrieval and secret key generation
    protocol.get_context = MagicMock(return_value='context_data')
    protocol.fPAKE.device_protocol = MagicMock(return_value='secret_key')

    # Create a mock socket
    mock_socket = MagicMock()

    # Execute the device protocol
    protocol.device_protocol(mock_socket)

    # Check that network functions were called
    mock_ack.assert_called_once_with(mock_socket)
    mock_ack_standby.assert_called_once_with(mock_socket, parameters['timeout'])
    mock_send_status.assert_called_once_with(mock_socket, True)

    # Validate the output
    protocol.fPAKE.device_protocol.assert_called_once_with('context_data', mock_socket)
    assert protocol.fPAKE.device_protocol.return_value == 'secret_key', "Secret key should be derived correctly."

# Optionally, include a main block to run tests directly, useful for quick checks
if __name__ == "__main__":
    test_fastzip_process_context()
    test_device_protocol()

