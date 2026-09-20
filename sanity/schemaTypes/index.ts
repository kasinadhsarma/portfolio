import { type SchemaTypeDefinition } from 'sanity'
import project from './project'
import certificate from './certificate'
import achievement from './achievement'
import skillCategory from './skillCategory'
import resumeFile from './resumeFile'
import resume from './resume'

export const schema: { types: SchemaTypeDefinition[] } = {
  types: [project, certificate, achievement, skillCategory, resumeFile, resume],
}
