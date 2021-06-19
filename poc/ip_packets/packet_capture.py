from pcapfile import savefile
from pcapfile.protocols.linklayer import ethernet
from pcapfile.protocols.network import ip
from pcapfile.protocols.transport import tcp
import binascii
import os
import sys

if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    pcap_file = savefile.load_savefile(open(sys.argv[1], 'rb'), verbose=True)

    f = open("/tmp/parsed.txt", "w")
    for count, pkt in enumerate(pcap_file.packets):
        print("\n**************************** Packet Number {} *****************************".format(count))
        packets_str = "Packet Length={} Capture Length={} Timestamp={}".format(pkt.packet_len, pkt.capture_len,
                                                                               pkt.timestamp)
        print(packets_str)

        eth_frame = ethernet.Ethernet(pkt.raw())
        ethernet_frame_str = "Eth Frame SRC={}, Eth Frame DST={}, Eth Frame Type={}".format(eth_frame.src,
                                                                                            eth_frame.dst,
                                                                                            eth_frame.type)
        print(ethernet_frame_str)
        if eth_frame.type == 2054:  # This is ARP message
            print("Not IPv4. Will skip...")
            continue
        ip_packet = ip.IP(binascii.unhexlify(eth_frame.payload))
        ip_packet_str = "IP Ver={} Internet Header Length={} Type of Service={} Total Len={} Identification={} " \
                        "Flags={} Fragment Offset={} TTL={} Protocol={} Header Checksum={} IP SRC={} IP DST={} Opt={}" \
                        " Pad={} Opt Parsed={}".format(ip_packet.v, ip_packet.hl, ip_packet.tos, ip_packet.len,
                                                       ip_packet.id, ip_packet.flags, ip_packet.off, ip_packet.ttl,
                                                       ip_packet.p, ip_packet.sum, ip_packet.src, ip_packet.dst,
                                                       ip_packet.opt, ip_packet.pad, ip_packet.opt_parsed)
        tcp_packet = tcp.TCP(ip_packet.payload)
        tcp_packet_str = "TCP Src Port={}, Dst Port={}, Seq No={}, Ack No={}, Data Offset={}, Reserved={}, Flags={}, " \
                         "Window={}, Checksum={}, Urgent Pointer={} Options={}, Padding={}, Data={}"\
            .format(tcp_packet.src_port, tcp_packet.dst_port, tcp_packet.seqnum, tcp_packet.acknum,
                    tcp_packet.data_offset, tcp_packet.rst, tcp_packet.fin, tcp_packet.win, tcp_packet.sum,
                    tcp_packet.urg, tcp_packet.opt, tcp_packet.psh, tcp_packet.payload)
        print(tcp_packet_str)
        print(ip_packet_str)
        f.write("# " + str(count) + " " + packets_str + "\n" + ethernet_frame_str + "\n" + ip_packet_str + "\n\n")
    f.close()
