def test_needsfiles(tmpdir):
    print ("Print Tmp Directory {}".format(tmpdir))


def test_function(record_xml_property):
    record_xml_property("example_key", 1)
    assert 1
