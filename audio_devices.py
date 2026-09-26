"""Windows endpoint format inspection for capture diagnostics."""
SPEAKERS = ('FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'FLC', 'FRC', 'BC', 'SL', 'SR',
            'TC', 'TFL', 'TFC', 'TFR', 'TBL', 'TBC', 'TBR')
SURROUND_MASKS = (0x3f, 0x60f, 0x63f)


def channel_labels(channels, mask):
    labels = [name for bit, name in enumerate(SPEAKERS) if mask & (1 << bit)]
    if len(labels) != channels:
        return [f'Ch {i + 1}' for i in range(channels)]
    return labels


def endpoint_format(device):
    # ponytail: SoundCard has no public channel-mask API. Reuse its WASAPI
    # bindings; replace this helper if an upgraded SoundCard changes internals.
    from soundcard.mediafoundation import _ffi, _com, _ole32
    client = device._audio_client()
    fmt = _ffi.new('WAVEFORMATEXTENSIBLE **')
    try:
        _com.check_error(client[0][0].lpVtbl.GetMixFormat(client[0], fmt))
        base = fmt[0].Format
        channels = int(base.nChannels)
        mask = int(fmt[0].dwChannelMask) if base.wFormatTag == 0xfffe and base.cbSize >= 22 else 0
        return {'channels': channels, 'rate': int(base.nSamplesPerSec), 'mask': mask,
                'labels': channel_labels(channels, mask)}
    finally:
        if fmt[0] != _ffi.NULL:
            _ole32.CoTaskMemFree(fmt[0])
        _com.release(client)
