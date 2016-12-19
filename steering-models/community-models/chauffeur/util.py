import logging, os, subprocess, sys, tempfile, time

logger = logging.getLogger(__name__)


def aws_credentials():
    """
    Get aws (key, secret)
    """
    return (
        os.environ['AWS_ACCESS_KEY_ID'],
        os.environ['AWS_SECRET_ACCESS_KEY'],
    )


def parse_s3_uri(s3_uri):
    """
    Parse a s3 uri into (bucket, key).

    @param s3_uri - formatted s3://bucket/key/path
    @return - (bucket, key)
    """
    assert s3_uri.startswith('s3://')
    return s3_uri.split('s3://')[-1].split('/', 1)


def upload_file(local_file, s3_uri):
    """
    Upload a local file to s3.

    @param local_file - local path to file to upload
    @param s3_uri - formatted s3://bucket/key/path
    """
    # use awscli for extra speed
    subprocess.call(['aws', 's3', 'cp', local_file, s3_uri])


def upload_dir(local_path, s3_uri):
    """
    Upload a local directory (bundled in .tar.gz) to s3.

    @param local_path - local directory path to upload
    @param s3_uri - formatted s3://bucket/dir/path
    """
    archive_path = local_path.rstrip('/') + '.tar.gz'
    s3_uri = s3_uri.rstrip('/').rstrip('.tar.gz') + '.tar.gz'

    logger.info('Archiving %s for upload', local_path)
    subprocess.call([
        'tar',
        'czf', archive_path,
        '-C', local_path,
        '.'])

    logger.info('Uploading %s to %s', archive_path, s3_uri)
    upload_file(archive_path, s3_uri)


def download_file(s3_bucket, s3_key, out_path):
    """
    Download a file from s3.

    @param s3_bucket - s3 bucket name
    @param s3_key - s3 key's path
    @param out_path - local download location
    """
    s3_uri = 's3://%s/%s' % (s3_bucket, s3_key)

    logger.info('Downloading ' + s3_uri)
    # use awscli for extra speed
    subprocess.call(['aws', 's3', 'cp', s3_uri, out_path])

def get_archive_s3_uri(s3_uri):
    """
    Get the archive s3 path for a s3 uri.

    @param - any s3 uri
    @return - uri with .tar.gz appended
    """
    s3_bucket, s3_key = parse_s3_uri(s3_uri)
    s3_key = s3_key.rstrip('/').rstrip('.tar.gz') + '.tar.gz'
    return 's3://%s/%s' % (s3_bucket, s3_key)

def download_dir(s3_uri, local_path):
    """
    Download an archived (.tar.gz) directory from s3.

    @param s3_uri - formatted s3://bucket/key/path
    @param local_path - local path to unpack archive to
    """
    s3_uri = get_archive_s3_uri(s3_uri)
    logger.info('Downloading and unarchiving %s to %s',
                s3_uri, local_path)

    try: os.makedirs(local_path)
    except: pass

    assert s3_uri.endswith('.tar.gz')
    _, tmp_path = tempfile.mkstemp()
    try:
        s3_bucket, s3_key = parse_s3_uri(s3_uri)
        download_file(s3_bucket, s3_key, tmp_path)

        # use awscli for faster download
        subprocess.call(['tar', 'xzf', tmp_path, '-C', local_path])


    finally:
        os.remove(tmp_path)
