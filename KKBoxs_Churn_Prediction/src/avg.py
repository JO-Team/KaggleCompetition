
import pandas as pd
import os

def avg_all_files(result_folder='output/'):

        if not os.path.exists(result_folder):
            raise Exception('Make sure {} is a correctly assigned path for files'.format(result_folder))

        sample_df = pd.read_csv('input/sample_submission_v2.csv')
        msno = sample_df['msno'].values

        targets = 0

        total = [content for content in os.listdir(result_folder) if 'csv' in content]
        siz = len(total)
        print('Average {} files'.format(siz))
        print(total)
        if siz == 0:
            raise Exception('Make sure you have put files under {}'.format(result_folder))


        for filename in total:
            fullpath = result_folder+filename
            print(fullpath)
            if filename.endswith('gz'):
                df = pd.read_csv(fullpath, compression='gzip')
            else:
                df = pd.read_csv(fullpath)

            print(df.info())
            print(df.is_churn.head())
            targets += df['is_churn']/siz


        avg = pd.DataFrame({'msno': msno,
                            'is_churn': targets})
        print(avg.head())

        filename = '{}/avg/avg_{}.csv'.format(result_folder, siz)
        if not os.path.exists('{}/avg/'.format(result_folder)):
            os.makedirs('{}/avg'.format(result_folder))

        avg.to_csv(filename, index=False)
        print('Generated ', filename)
        return avg


if __name__ == '__main__':

    avg_all_files('final/')
