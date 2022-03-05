import re
import argparse
import pyshark
import sys, os.path
import matplotlib.pyplot as plt
# from matplotlib.externals import six
import numpy as np
from matplotlib.ticker import FuncFormatter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pcap_files", dest="pcaps", help="Specify one or more PCAP files", nargs='+',
                        required=True)
    return parser.parse_args()


def get_source(ip_packet):
    if hasattr(ip_packet, 'ip'):
        return ip_packet.ip.src
    elif hasattr(ip_packet, 'ipv6'):
        return ip_packet.ipv6.src
    else:
        return None


def get_destination(ip_packet):
    if hasattr(ip_packet, 'ip'):
        return ip_packet.ip.dst
    elif hasattr(ip_packet, 'ipv6'):
        return ip_packet.ipv6.dst
    else:
        return None


def get_packet_size(ip_packet):
    if hasattr(ip_packet, 'captured_length'):
        return ip_packet.captured_length


def filter_packets(pcap):
    _ip_packets = []
    _pattern = re.compile('10\\.1\\.1\\.[4|5]')
    for _ip_packet in pcap:
        _src_ip = get_source(_ip_packet)
        _dst_ip = get_destination(_ip_packet)
        if _src_ip is None or _dst_ip is None:
            continue
        if _pattern.match(_src_ip) is None or _pattern.match(_dst_ip) is None:
            continue
        _ip_packets.append(_ip_packet)
    return _ip_packets


if __name__ == '__main__':
    # Total Network Traffic
    # Frequency of HeartBeats, Count of Packets over period
    # Total Bytes Transferred, Min, Max, Avg Size of Packets
    #
    args = arg_parse()
    for _pcap_file in args.pcaps:
        _packets = pyshark.FileCapture(_pcap_file, keep_packets=False)
        # _packets = filter_packets(_packets)
        _tcp_count = 0
        _ucp_count = 0
        _others_count = 0
        _total_count = 0
        _total_bytes = 0
        _avg_bytes = 0
        _min_size_bytes = 10000000000
        _max_size_bytes = 0
        _packet_sizes = []
        _packet_avg_sizes = []

        for _packet in _packets:
            if hasattr(_packet, 'tcp'):
                _tcp_count = _tcp_count + 1
            elif hasattr(_packet, 'udp'):
                _ucp_count = _ucp_count + 1
            else:
                _others_count = _others_count + 1
            _total_count = _total_count + 1
            _size = int(get_packet_size(_packet))

            _total_bytes = _total_bytes + _size
            _avg_bytes = _total_bytes // _total_count

            _packet_sizes.append(_size)
            _packet_avg_sizes.append(_avg_bytes)

            _min_size_bytes = min(_min_size_bytes, _size)
            _max_size_bytes = max(_max_size_bytes, _size)

            print("src={} dst={} size={} running_avg={}".format(get_source(_packet), get_destination(_packet),
                                                                _size, _avg_bytes))
            if _total_count == 1000:
                break
        # End of For loop

        X_Value = np.linspace(1, _total_count, _total_count)
        print("TCP Count={}, UDP Count={}, Others Count={}, Total Count={}, Total Bytes={} Avg Bytes={}"
              .format(_tcp_count, _ucp_count, _others_count, _total_count, _total_bytes, _avg_bytes))

        # fig, ax = plt.subplots()
        # ax.plot(X_Value, np.array(_packet_sizes))
        # ax.plot(X_Value, np.array(_packet_avg_sizes))

        plt.title('Packet information Graph')
        plt.xlabel('Relative Time')
        plt.ylabel('Packet Size (Bytes)')
        plt.ylim(bottom=0, top=_max_size_bytes + 100)
        plt.plot(X_Value, np.array(_packet_sizes), label="Pkt Sizes")
        plt.plot(X_Value, np.array(_packet_avg_sizes), label="Running Avg Pkt Sizes")
        leg = plt.legend(loc='upper right')
        plt.show()
