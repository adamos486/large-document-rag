'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { cn } from '@/lib/utils'
import { FileQueue } from './file-queue'

// Supported financial document types
const ACCEPTED_FILE_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-excel': ['.xls'],
  'text/csv': ['.csv'],
  'application/json': ['.json'],
  'text/plain': ['.txt'],
}

// API endpoint for file uploads
const UPLOAD_ENDPOINT = 'http://localhost:8000/api/upload'

// File statuses
type FileStatus = 'pending' | 'uploading' | 'success' | 'error'

// File with upload status information
interface FileWithStatus {
  id: string
  file: File
  status: FileStatus
  progress: number
  error?: string
  taskId?: string // Store the task ID returned from the API
}

// Response from the upload API
interface UploadResponse {
  task_id: string
  file_name: string
  collection_name: string
  status: string
}

export function FileUploader() {
  const [files, setFiles] = useState<FileWithStatus[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [collectionName, setCollectionName] = useState('default') // Default collection name
  const { toast } = useToast()

  // Handle file drop
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const newFiles = acceptedFiles.map((file) => ({
        id: crypto.randomUUID(), //Should the id be the file name?
        file,
        status: 'pending' as FileStatus,
        progress: 0,
      }))

      setFiles((prev) => [...prev, ...newFiles])

      toast({
        title: `${acceptedFiles.length} file${acceptedFiles.length > 1 ? 's' : ''} added`,
        description: 'Ready to upload',
      })
    },
    [toast],
  )

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_FILE_TYPES,
    maxSize: 10485760, // 10MB
    onDropRejected: (rejections) => {
      toast({
        variant: 'destructive',
        title: 'File upload failed',
        description: 'Please check file type and size (max 10MB)',
      })
    },
  })

  // Process files in queue
  const processFiles = async () => {
    if (files.length === 0 || isUploading) return

    setIsUploading(true)

    // Process files one by one
    for (const fileItem of files.filter((f) => f.status === 'pending')) {
      // Update status to uploading
      setFiles((prev) =>
        prev.map((f) => (f.id === fileItem.id ? { ...f, status: 'uploading' as FileStatus } : f)),
      )

      try {
        // Perform actual file upload
        await uploadFile(fileItem)

        // Mark as success
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileItem.id ? { ...f, status: 'success' as FileStatus, progress: 100 } : f,
          ),
        )
      } catch (error) {
        // Mark as error
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileItem.id
              ? {
                  ...f,
                  status: 'error' as FileStatus,
                  error: error instanceof Error ? error.message : 'Upload failed',
                }
              : f,
          ),
        )
      }
    }

    setIsUploading(false)

    // Check if all files are processed
    const allProcessed = files.every((f) => f.status === 'success' || f.status === 'error')
    const successCount = files.filter((f) => f.status === 'success').length

    if (allProcessed && successCount > 0) {
      toast({
        title: 'Upload complete',
        description: `Successfully processed ${successCount} of ${files.length} files`,
      })
    }
  }

  // Upload file to backend API
  const uploadFile = async (fileItem: FileWithStatus): Promise<void> => {
    // Create FormData object
    const formData = new FormData()
    formData.append('collection_name', collectionName)
    formData.append('file', fileItem.file)
    formData.append('custom_metadata', JSON.stringify({}))

    // Don't add collection_name as query parameter since we're already sending it in the form
    const url = UPLOAD_ENDPOINT

    // Note: Fetch API doesn't support upload progress natively
    // We'll update progress at specific points instead

    // Set initial progress
    setFiles((prev) => prev.map((f) => (f.id === fileItem.id ? { ...f, progress: 10 } : f)))

    // Make the fetch request - don't set Content-Type header
    // Let the browser set the correct multipart/form-data with boundary
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    })

    // Update progress to 90% after fetch completes but before processing response
    setFiles((prev) => prev.map((f) => (f.id === fileItem.id ? { ...f, progress: 90 } : f)))

    // Handle response
    if (!response.ok) {
      throw new Error(`Upload failed with status: ${response.status}`)
    }

    // Parse response
    const responseData: UploadResponse = await response.json()

    // Store task ID for potential status checking
    setFiles((prev) =>
      prev.map((f) =>
        f.id === fileItem.id
          ? {
              ...f,
              taskId: responseData.task_id,
              progress: 100,
            }
          : f,
      ),
    )
  }

  // Remove a file from the queue
  const removeFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId))
  }

  // Retry a failed upload
  const retryFile = (fileId: string) => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === fileId ? { ...f, status: 'pending', progress: 0, error: undefined } : f,
      ),
    )
  }

  // Clear all completed files
  const clearCompleted = () => {
    setFiles((prev) => prev.filter((f) => f.status !== 'success'))
  }

  return (
    <div className='space-y-6'>
      {/* Collection selection */}
      <div className='space-y-2'>
        <label htmlFor='collection-name' className='text-sm font-medium'>
          Collection Name
        </label>
        <div className='flex space-x-2'>
          <input
            id='collection-name'
            type='text'
            value={collectionName}
            onChange={(e) => setCollectionName(e.target.value)}
            className='flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50'
            placeholder='Enter collection name'
          />
        </div>
      </div>

      {/* Dropzone */}
      <Card
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed p-10 text-center cursor-pointer transition-colors',
          isDragActive
            ? 'border-primary bg-primary/5'
            : 'border-muted-foreground/25 hover:border-primary/50',
        )}
      >
        <input {...getInputProps()} />
        <div className='flex flex-col items-center justify-center space-y-4'>
          <div className='rounded-full bg-primary/10 p-4'>
            <Upload className='h-8 w-8 text-primary' />
          </div>
          <div className='space-y-2'>
            <h3 className='text-lg font-semibold'>
              {isDragActive ? 'Drop files here' : 'Drag & drop files'}
            </h3>
            <p className='text-sm text-muted-foreground max-w-md mx-auto'>
              Drop your financial documents here or click to browse. Supports PDF, Excel, CSV, JSON,
              and TXT files (max 10MB).
            </p>
          </div>
          <Button variant='outline' type='button' disabled={isDragActive}>
            Browse files
          </Button>
        </div>
      </Card>
      {/* File Queue */}
      {files.length > 0 && (
        <div className='space-y-4'>
          <div className='flex items-center justify-between'>
            <h3 className='text-lg font-semibold'>Files ({files.length})</h3>
            <div className='flex gap-2'>
              <Button
                variant='outline'
                size='sm'
                onClick={clearCompleted}
                disabled={!files.some((f) => f.status === 'success')}
              >
                Clear completed
              </Button>
              <Button
                onClick={processFiles}
                disabled={isUploading || !files.some((f) => f.status === 'pending')}
                size='sm'
              >
                {isUploading ? 'Processing...' : 'Process all'}
              </Button>
            </div>
          </div>

          <FileQueue files={files} onRemove={removeFile} onRetry={retryFile} />
        </div>
      )}
    </div>
  )
}
