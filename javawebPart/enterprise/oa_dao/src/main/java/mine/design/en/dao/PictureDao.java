package mine.design.en.dao;
import mine.design.en.entity.Picture;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository("pictureDao")
public interface PictureDao {
    void insert (Picture picture);
    List<Picture> selectTime();
    List<Picture> selectAccurate();
    List<Picture> getAll();

}
