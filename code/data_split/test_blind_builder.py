import unittest
import pandas as pd
from blind_builder import build,FIELDS

class BoundaryTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([['W00',str(i),'p'+str(i//4),'sp','h','["'+str(i)+'"]','50',str(i%4)] for i in range(40)],columns=FIELDS)
    def test_shuffle_invariance_and_parent_closure(self):
        d=self.data(); a=build(d,'S3_PARENT'); b=build(d.sample(frac=1,random_state=4),'S3_PARENT')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.groupby('canonical_parent').assigned_split.nunique().max(),1)
    def test_source_transitive_closure(self):
        d=self.data(); d.loc[0,'source_result_ids_json']='["a","b"]'; d.loc[8,'source_result_ids_json']='["b","c"]'; d.loc[20,'source_result_ids_json']='["c"]'
        a=build(d,'S1_CONDITION').set_index('stable_record_id')
        self.assertEqual(len(set(a.loc[['0','8','20'],'assigned_split'])),1)
    def test_reject_labels_and_original(self):
        d=self.data(); d['target']=0
        with self.assertRaises(ValueError): build(d,'S2_COMBINATION')
        with self.assertRaises(ValueError): build(self.data(),'S0_ORIGINAL')
    def test_route_independent(self):
        d=self.data(); other=d.copy(); other['route']='M00'
        a=build(d,'S2_COMBINATION'); b=build(pd.concat([d,other]),'S2_COMBINATION')
        pd.testing.assert_frame_equal(a,b[b.route=='W00'].reset_index(drop=True))
    def test_source_test_closure_only_in_triggered_split(self):
        d=self.data();d['test_ids']=['["t'+str(i//8)+'"]' for i in range(len(d))]
        a=build(d,'S_SOURCE')
        self.assertEqual(a.groupby('test_ids').assigned_split.nunique().max(),1)
        with self.assertRaises(ValueError):build(d,'S1_CONDITION')
    def test_multitime_condition_refused(self):
        d=self.data();d.loc[0,'condition_time_key']='["0x1.0p+0","0x1.0p+1"]'
        with self.assertRaises(ValueError):build(d,'S1_CONDITION')
        self.assertEqual(len(build(d,'S3_PARENT')),len(d))

if __name__=='__main__': unittest.main()
