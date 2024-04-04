import os

from cryptography.hazmat.primitives import constant_time, hmac

from network import *

def send_nonce_msg_to_device(
    connection, recieved_nonce_msg, derived_key, prederived_key_hash, nonce_byte_size, hash_func,
):
    nonce = os.urandom(nonce_byte_size)

    # Concatenate nonces together
    pd_hash_len = len(prederived_key_hash)
    recieved_nonce = recieved_nonce_msg[
        pd_hash_len : pd_hash_len + nonce_byte_size
    ]
    concat_nonce = nonce + recieved_nonce

    # Create tag of Nonce
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(concat_nonce)
    tag = mac.finalize()

    # Construct nonce message
    nonce_msg = nonce + tag

    send_nonce_msg(connection, nonce_msg)

    return nonce

def send_nonce_msg_to_host(connection, prederived_key_hash, derived_key, nonce_byte_size, hash_func,):
    # Generate Nonce
    nonce = os.urandom(nonce_byte_size)

    # Create tag of Nonce
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(nonce)
    tag = mac.finalize()

    # Create key confirmation message
    nonce_msg = prederived_key_hash + nonce + tag

    send_nonce_msg(connection, nonce_msg)

    return nonce

def verify_mac_from_host(recieved_nonce_msg, generated_nonce, derived_key, nonce_byte_size, hash_func,):
    success = False

    recieved_nonce = recieved_nonce_msg[0 : nonce_byte_size]

    # Create tag of Nonce
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(recieved_nonce + generated_nonce)
    generated_tag = mac.finalize()

    recieved_tag = recieved_nonce_msg[nonce_byte_size :]
    if constant_time.bytes_eq(generated_tag, recieved_tag):
        success = True
    return success

def verify_mac_from_device(
    recieved_nonce_msg, derived_key, prederived_key_hash, nonce_byte_size, hash_func,
):
    success = False

    # Retrieve nonce used by device
    pd_hash_len = len(prederived_key_hash)
    recieved_nonce = recieved_nonce_msg[
        pd_hash_len : pd_hash_len + nonce_byte_size
    ]

    # Generate new MAC tag for the nonce with respect to the derived key
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(recieved_nonce)
    generated_tag = mac.finalize()

    recieved_tag = recieved_nonce_msg[pd_hash_len + nonce_byte_size :]
    if constant_time.bytes_eq(generated_tag, recieved_tag):
        success = True
    return success