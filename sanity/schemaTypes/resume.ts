import { CaseIcon } from '@sanity/icons'
import { defineArrayMember, defineField, defineType } from 'sanity'

export default defineType({
  name: 'resume',
  title: 'Resume',
  type: 'document',
  icon: CaseIcon,
  fields: [
    defineField({
      name: 'education',
      title: 'Education',
      type: 'array',
      of: [
        defineArrayMember({
          type: 'object',
          name: 'educationEntry',
          fields: [
            { name: 'institution', title: 'Institution', type: 'string' },
            { name: 'period', title: 'Period', type: 'string' },
            { name: 'description', title: 'Description', type: 'text', rows: 3 },
          ],
          preview: {
            select: { title: 'institution', subtitle: 'period' },
          },
        }),
      ],
    }),
    defineField({
      name: 'experience',
      title: 'Professional Experience',
      type: 'array',
      of: [
        defineArrayMember({
          type: 'object',
          name: 'experienceEntry',
          fields: [
            { name: 'title', title: 'Title', type: 'string' },
            { name: 'organization', title: 'Organization', type: 'string' },
            { name: 'period', title: 'Period', type: 'string' },
            { name: 'current', title: 'Current', type: 'boolean', initialValue: false },
            {
              name: 'highlights',
              title: 'Highlights',
              type: 'array',
              of: [{ type: 'string' }],
            },
          ],
          preview: {
            select: { title: 'title', subtitle: 'organization' },
          },
        }),
      ],
    }),
    defineField({
      name: 'projects',
      title: 'Projects',
      type: 'array',
      of: [
        defineArrayMember({
          type: 'object',
          name: 'resumeProjectEntry',
          fields: [
            { name: 'title', title: 'Title', type: 'string' },
            { name: 'subtitle', title: 'Subtitle', type: 'string' },
            {
              name: 'highlights',
              title: 'Highlights',
              type: 'array',
              of: [{ type: 'string' }],
            },
          ],
          preview: {
            select: { title: 'title', subtitle: 'subtitle' },
          },
        }),
      ],
    }),
    defineField({
      name: 'training',
      title: 'Training & Internships',
      type: 'array',
      of: [
        defineArrayMember({
          type: 'object',
          name: 'trainingEntry',
          fields: [
            { name: 'title', title: 'Title', type: 'string' },
            { name: 'organization', title: 'Organization', type: 'string' },
            { name: 'period', title: 'Period', type: 'string' },
            { name: 'description', title: 'Description', type: 'text', rows: 3 },
          ],
          preview: {
            select: { title: 'title', subtitle: 'organization' },
          },
        }),
      ],
    }),
  ],
  preview: {
    prepare() {
      return { title: 'Resume' }
    },
  },
})
