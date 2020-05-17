from jira import JIRA

user = 'peter@impact.com'
apikey = None
password = None
server = 'https://'

jira = JIRA(basic_auth=("peter@impact.com", apikey), options={'server': server})

# ticket = 'MIDS-999'
# issue = jira.issue(ticket)
#
# summary = issue.fields.summary
#
# print('ticket: ', ticket, summary)

####

# jira.create_issue(project='MIDS', summary='QA Recommendations Test',
#                               description='QA Recommendations Test', issuetype={'name': 'Bug'})

####

# Find issues with attachments:
query = jira.search_issues(jql_str="""project = IRTS
AND created > -2w
and assignee != EMPTY
and assignee = 557058:c1917ccb-e4c2-4bf8-aa5a-01f87082e627
 """, json_result=True, fields="key, attachment")

print("Test")
print(query)
# And remove attachments one by one
for i in query['issues']:
    for a in i['fields']['attachment']:
        print("For issue {0}, found attach: '{1}' [{2}].".format(i['key'], a['filename'], a['id']))
