"""List VPD checkpoint objects under the run prefix (keys, sizes)."""
from common import R2_PREFIX, credentials, s3
credentials(); client = s3(); paginator = client.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket='param-clustering', Prefix=R2_PREFIX):
    for obj in page.get('Contents', []):
        k = obj['Key']
        if k.endswith(('.pth', '.pt')) and 'vpd' in k.lower(): print(f"{obj['Size']/1e9:6.2f} GB  {k.replace(R2_PREFIX+'/', '')}", flush=True)
print('R2_LIST_DONE', flush=True)
