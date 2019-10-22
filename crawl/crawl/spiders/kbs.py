import scrapy


class KBSSpider(scrapy.Spider):
    """
    Scrapes the KBS website to obtain raw Korean text from their online news articles archives.
    """
    # Assign the spider a unique id
    name = "kbs"

    def start_requests(self):
        """
        Set the seed KBS url to begin scraping with
        KBS archives their articles with a running index - the larger the id, the newer the article
        For our purposes we start from `root_url` and scrape backwards into older articles, see parse()
        :return: An iterable of Requests (list or generator)
        """
        root_url = 'http://news.kbs.co.kr/news/view.do?ncd=4282000'
        yield scrapy.Request(url=root_url, callback=self.parse)

    def parse(self, response):
        """
        Parses each retrieved html response to extract the article's title and body text and save it to a file
        :param response: An instance of TextResponse containing the retrieved page content
        :return: A Request instance to follow a specified link
        """

        # prepend pageid with 0s if necessary to make it a seven digit value
        zfill = 7

        baseurl = response.url.split("ncd=")[0] + "ncd="
        pageid = response.url.split("ncd=")[-1]

        # where to save the extracted text to
        filetext = 'hangul.txt'.format(pageid)

        if response.status == 200:
            # extract title from news article
            title = response.css('title::text').get()
            if title is not None and '>' in title:
                title = title.split('>')[0]
            else:
                title = '--**--'
            # extract body text from news article
            content = response.xpath('//div[@class="detail-body font-size"]//text()').getall()
            if content is not None:
                str_content = ''
                for elem in content:
                    str_content += elem.replace('\n', '').replace('\t', '')
            else:
                str_content = '--**--'

            # save to file
            with open(filetext, 'a') as f:
                f.write('{} {}\n{}\n\n'.format(pageid, title, str_content))
            self.log('Wrote article contents to %s' % filetext)

        # continue scraping next article if available
        nextid = None
        if int(pageid) > 1:
            nextid = baseurl + str(int(pageid)-1).zfill(zfill)
        if nextid is not None:
            yield response.follow(nextid, callback=self.parse)

