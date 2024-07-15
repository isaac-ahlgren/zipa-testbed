import os

from cryptography.hazmat.primitives import constant_time, hmac
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from networking.network import dh_exchange, dh_exchange_standby, send_nonce_msg

ec_curve = ec.SECP384R1()


def send_nonce_msg_to_device(
    connection,
    recv_nonce_msg,
    derived_key,
    prederived_key_hash,
    nonce_byte_size,
    hash_func,
):
    nonce = os.urandom(nonce_byte_size)

    # Concatenate nonces together
    pd_hash_len = len(prederived_key_hash)
    recv_nonce = recv_nonce_msg[pd_hash_len : pd_hash_len + nonce_byte_size]
    concat_nonce = nonce + recv_nonce

    # Create tag of Nonce
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(concat_nonce)
    tag = mac.finalize()

    # Construct nonce message
    nonce_msg = nonce + tag

    send_nonce_msg(connection, nonce_msg)

    return nonce


def send_nonce_msg_to_host(
    connection,
    prederived_key_hash,
    derived_key,
    nonce_byte_size,
    hash_func,
):
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


def verify_mac_from_host(
    recieved_nonce_msg,
    generated_nonce,
    derived_key,
    nonce_byte_size,
    hash_func,
):
    success = False

    recieved_nonce = recieved_nonce_msg[0:nonce_byte_size]

    # Create tag of Nonce
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(recieved_nonce + generated_nonce)
    generated_tag = mac.finalize()

    recieved_tag = recieved_nonce_msg[nonce_byte_size:]
    if constant_time.bytes_eq(generated_tag, recieved_tag):
        success = True
    return success


def verify_mac_from_device(
    recv_nonce_msg,
    derived_key,
    prederived_key_hash,
    nonce_byte_size,
    hash_func,
):
    success = False

    # Retrieve nonce used by device
    pd_hash_len = len(prederived_key_hash)
    recv_nonce = recv_nonce_msg[pd_hash_len : pd_hash_len + nonce_byte_size]

    # Generate new MAC tag for the nonce with respect to the derived key
    mac = hmac.HMAC(derived_key, hash_func)
    mac.update(recv_nonce)
    generated_tag = mac.finalize()

    recieved_tag = recv_nonce_msg[pd_hash_len + nonce_byte_size :]
    if constant_time.bytes_eq(generated_tag, recieved_tag):
        success = True
    return success


def diffie_hellman(socket, timeout=30, verbose=True):
    # Generate initial private key for Diffie-Helman
    initial_private_key = ec.generate_private_key(ec_curve)

    public_key = initial_private_key.public_key().public_bytes(
        Encoding.X962, PublicFormat.CompressedPoint
    )

    # Send initial key for Diffie-Helman
    if verbose:
        print("Send DH public key\n")

    dh_exchange(socket, public_key)

    # Recieve other devices key
    if verbose:
        print("Waiting for DH public key\n")

    other_public_key_bytes = dh_exchange_standby(socket, timeout)

    if other_public_key_bytes is None:
        if verbose:
            print("No initial key for Diffie-Helman recieved - early exit\n")
        return

    other_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec_curve, other_public_key_bytes
    )

    # Shared key generated
    shared_key = initial_private_key.exchange(ec.ECDH(), other_public_key)

    return shared_key
