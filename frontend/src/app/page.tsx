import { FileUploader } from '@/components/file-uploader'
import { PageHeader } from '@/components/page-header'

export default function Home() {
  return (
    <div className='container mx-auto py-10 space-y-8'>
      <PageHeader
        title='Financial Document Upload'
        description='Upload and process multiple financial documents at once'
      />
      <FileUploader />
    </div>
  )
}
