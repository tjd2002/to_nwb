from datetime import datetime

from pynwb import NWBFile

mock_nwb_file = NWBFile(source='source',
                        session_description='session_description',
                        identifier='identifier',
                        session_start_time=datetime.now(),
                        file_create_date=datetime.now(),
                        institution='institution',
                        lab='lab')

