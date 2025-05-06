'use client'

import {
  FileText,
  FileIcon as FilePdf,
  FileSpreadsheet,
  FileX,
  AlertCircle,
  X,
  RefreshCw,
} from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

// File statuses
type FileStatus = 'pending' | 'uploading' | 'success' | 'error'

// File with upload status information
interface FileWithStatus {
  id: string
  file: File
  status: FileStatus
  progress: number
  error?: string
}

interface FileQueueProps {
  files: FileWithStatus[]
  onRemove: (id: string) => void
  onRetry: (id: string) => void
}

export function FileQueue({ files, onRemove, onRetry }: FileQueueProps) {
  // Get appropriate icon based on file type
  const getFileIcon = (file: File) => {
    const type = file.type

    if (type === 'application/pdf') {
      return <FilePdf className='h-5 w-5 text-red-500' />
    } else if (
      type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
      type === 'application/vnd.ms-excel' ||
      type === 'text/csv'
    ) {
      return <FileSpreadsheet className='h-5 w-5 text-green-600' />
    } else if (type === 'application/json' || type === 'text/plain') {
      return <FileText className='h-5 w-5 text-blue-500' />
    } else {
      return <FileX className='h-5 w-5 text-gray-500' />
    }
  }

  // Get status badge
  const getStatusBadge = (status: FileStatus) => {
    switch (status) {
      case 'pending':
        return <Badge variant='outline'>Pending</Badge>
      case 'uploading':
        return <Badge variant='secondary'>Uploading</Badge>
      case 'success':
        return (
          <Badge variant='success' className='bg-green-500 hover:bg-green-600'>
            Complete
          </Badge>
        )
      case 'error':
        return <Badge variant='destructive'>Failed</Badge>
    }
  }

  return (
    <div className='space-y-3'>
      {files.map((fileItem) => (
        <Card key={fileItem.id} className='p-4'>
          <div className='flex items-start gap-4'>
            <div className='flex-shrink-0'>{getFileIcon(fileItem.file)}</div>

            <div className='flex-1 min-w-0'>
              <div className='flex items-center justify-between mb-1'>
                <div className='flex items-center gap-2'>
                  <span className='font-medium truncate' title={fileItem.file.name}>
                    {fileItem.file.name}
                  </span>
                  {getStatusBadge(fileItem.status)}
                </div>

                <div className='flex items-center gap-2'>
                  {fileItem.status === 'error' && (
                    <Button
                      variant='ghost'
                      size='icon'
                      onClick={() => onRetry(fileItem.id)}
                      className='h-8 w-8'
                    >
                      <RefreshCw className='h-4 w-4' />
                      <span className='sr-only'>Retry</span>
                    </Button>
                  )}

                  <Button
                    variant='ghost'
                    size='icon'
                    onClick={() => onRemove(fileItem.id)}
                    className='h-8 w-8'
                  >
                    <X className='h-4 w-4' />
                    <span className='sr-only'>Remove</span>
                  </Button>
                </div>
              </div>

              <div className='flex items-center gap-2 text-xs text-muted-foreground mb-2'>
                <span>{(fileItem.file.size / 1024).toFixed(1)} KB</span>
                <span>•</span>
                <span>{fileItem.file.type.split('/')[1]?.toUpperCase() || 'Unknown'}</span>
              </div>

              {fileItem.error && (
                <div className='flex items-center gap-1 text-xs text-destructive mb-2'>
                  <AlertCircle className='h-3 w-3' />
                  <span>{fileItem.error}</span>
                </div>
              )}

              <Progress
                value={fileItem.progress}
                className={`h-2 ${
                  fileItem.status === 'success'
                    ? 'bg-muted/50'
                    : fileItem.status === 'error'
                      ? 'bg-destructive/20'
                      : 'bg-muted/50'
                }`}
                indicatorClassName={
                  fileItem.status === 'success'
                    ? 'bg-green-500'
                    : fileItem.status === 'error'
                      ? 'bg-destructive'
                      : undefined
                }
              />
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
